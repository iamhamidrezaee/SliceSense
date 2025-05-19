import os
import glob
import random
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.amp import autocast
import nibabel as nib
from scipy import stats

# ========================
# Model Definition
# ========================
class SequenceClassifier(nn.Module):
    """
    Neural network model for MRI slice thickness classification.
    Uses ResNet-18 backbone with BiLSTM and attention for sequence processing.
    """
    def __init__(self, feature_dim=256, hidden_dim=128, num_layers=2, num_classes=6):
        super(SequenceClassifier, self).__init__()
        # Pretrained ResNet-18 backbone
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.encoder = backbone
        
        # Project ResNet features to desired dimension
        self.fc_proj = nn.Sequential(
            nn.Linear(512, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()
        )

        # Bidirectional LSTM for sequence modeling
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Self-attention mechanism for sequence
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim*2,  # *2 for bidirectional
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # Final MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes)
        )
        
    def forward(self, x):
        B, S, C, H, W = x.shape
        
        # Extract features from each image in sequence
        x = x.view(B * S, C, H, W)
        feats = self.encoder(x)            # (B*S, 512)
        feats = self.fc_proj(feats)        # (B*S, feature_dim)
        feats = feats.view(B, S, -1)       # (B, S, feature_dim)

        # Process sequence with BiLSTM
        lstm_out, _ = self.lstm(feats)     # (B, S, 2*hidden_dim)
        
        # Apply self-attention to capture relationships between slices
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling (both average and max pooling for robustness)
        avg_pool = torch.mean(attn_out, dim=1)  # (B, 2*hidden_dim)
        max_pool, _ = torch.max(attn_out, dim=1)  # (B, 2*hidden_dim)
        pooled = avg_pool + 0.5 * max_pool  # Weighted combination
        
        # Classification
        return self.classifier(pooled)

# ========================
# Transform for inference
# ========================
MRI_MEAN, MRI_STD = [0.3, 0.3, 0.3], [0.15, 0.15, 0.15]

inference_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MRI_MEAN, MRI_STD)
])

# ========================
# Model Loading Function
# ========================
def load_model(model_path, device):
    """
    Load a trained model from disk.
    
    Args:
        model_path (str): Path to the model file
        device (torch.device): Device to load model onto
    
    Returns:
        nn.Module: Loaded model in evaluation mode
    """
    model = SequenceClassifier(feature_dim=128, hidden_dim=64, num_layers=1, num_classes=6)
    
    # Load state dict
    state_dict = torch.load(model_path, map_location=device, weights_only=False)
    
    # Handle compiled model state dict
    if all(key.startswith('_orig_mod.') for key in state_dict.keys()):
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace('_orig_mod.', '')
            new_state_dict[new_key] = value
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# ========================
# Sliding Window Inference
# ========================
def sliding_window_inference(model, image_paths, window_size=10, stride=5, device='cuda'):
    """
    Perform inference on a sequence of images using sliding window approach.
    
    Args:
        model (nn.Module): Trained model
        image_paths (list): List of image file paths
        window_size (int): Size of sliding window
        stride (int): Stride for sliding window
        device (str): Device to run inference on
    
    Returns:
        tuple: (predicted_class, class_probabilities)
    """
    model.eval()
    num_images = len(image_paths)
    predictions = []
    probabilities = []
    
    # Calculate number of windows to have odd number of predictions
    num_windows = 0
    for start_idx in range(0, num_images - window_size + 1, stride):
        num_windows += 1
    
    # Adjust stride if needed to make num_windows odd
    if num_windows % 2 == 0:
        stride = (num_images - window_size) // (num_windows - 1)
        if stride < 1:
            stride = 1
    
    with torch.no_grad():
        for start_idx in range(0, num_images - window_size + 1, stride):
            end_idx = start_idx + window_size
            window_paths = image_paths[start_idx:end_idx]
            
            # Load and transform images
            images = []
            for img_path in window_paths:
                img = Image.open(img_path).convert("RGB")
                img = inference_transforms(img)
                images.append(img)
            
            # Stack images into sequence
            sequence = torch.stack(images, dim=0).unsqueeze(0)  # (1, 10, C, H, W)
            sequence = sequence.to(device)
            
            # Perform inference
            with autocast(device_type='cuda' if device == 'cuda' else 'cpu'):
                outputs = model(sequence)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                pred = outputs.argmax(dim=1).item()
            
            predictions.append(pred)
            probabilities.append(probs.cpu().numpy()[0])
    
    # Majority voting
    majority_vote = stats.mode(predictions, keepdims=False)[0]
    avg_probabilities = np.mean(probabilities, axis=0)
    
    return majority_vote, avg_probabilities

# ========================
# 3D Reconstruction
# ========================
def reconstruct_3d(image_paths, predicted_thickness):
    """
    Reconstruct 3D NIfTI image from JPEGs with predicted slice thickness.
    
    Args:
        image_paths (list): List of image file paths
        predicted_thickness (int): Predicted thickness class
    
    Returns:
        nib.Nifti1Image: Reconstructed 3D NIfTI image
    """
    # Load all images and stack them
    images = []
    for img_path in sorted(image_paths):
        img = Image.open(img_path).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        images.append(img_array)
    
    # Stack to create 3D volume
    volume = np.stack(images, axis=-1)
    
    # Map class to thickness in mm
    thickness_map = {
        0: 0.5,
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 5.0,
        5: 7.0
    }
    
    # Create affine matrix with proper spacing
    thickness = thickness_map[predicted_thickness]
    affine = np.eye(4)
    # Pixel spacing is 1mm in x,y directions
    affine[0, 0] = 1.0  # x-axis spacing (mm)
    affine[1, 1] = 1.0  # y-axis spacing (mm)
    affine[2, 2] = thickness  # z-axis spacing (slice thickness)
    
    # Create NIfTI image
    nifti_img = nib.Nifti1Image(volume, affine)
    
    return nifti_img

# ========================
# Main API Function
# ========================
def slicesense_inference(model, input_path, output_path=None):
    """
    Perform inference on MRI slices to predict thickness and reconstruct 3D image.
    
    Args:
        model (str): Path to the trained model file
        input_path (str): Path to folder containing JPEG slices
        output_path (str, optional): Path to save the reconstructed NIfTI image.
                                   If None, creates 'slicesense_output' in current directory.
    
    Returns:
        str: Path to the saved NIfTI image
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Handle output directory
    if output_path is None:
        output_path = os.path.join(os.getcwd(), 'slicesense_output')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Load model with fix for compiled models
    model_obj = load_model(model, device)
    
    # Get all JPEG images in the folder
    image_paths = sorted(glob.glob(os.path.join(input_path, "*.jpg")))
    if not image_paths:
        raise ValueError(f"No JPEG images found in {input_path}")
    
    # Perform sliding window inference
    predicted_class, probabilities = sliding_window_inference(model_obj, image_paths, device=device)
    
    # Reconstruct 3D image
    nifti_img = reconstruct_3d(image_paths, predicted_class)
    
    # Save output
    folder_name = os.path.basename(input_path)
    output_filename = os.path.join(output_path, f"{folder_name}_thickness_{predicted_class}.nii.gz")
    nib.save(nifti_img, output_filename)
    
    print(f"Predicted thickness class: {predicted_class} ({get_thickness_mm(predicted_class)}mm)")
    print(f"Confidence: {probabilities[predicted_class]:.2%}")
    print(f"Saved to: {output_filename}")
    
    return output_filename

def get_thickness_mm(class_label):
    """
    Convert class label to actual thickness in mm.
    
    Args:
        class_label (int): Class label (0-5)
    
    Returns:
        float: Thickness in mm
    """
    thickness_map = {
        0: 0.5,
        1: 1.0,
        2: 2.0,
        3: 3.0,
        4: 5.0,
        5: 7.0
    }
    return thickness_map[class_label]