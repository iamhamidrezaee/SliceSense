import os
import glob
import random
import pandas as pd
import numpy as np
from PIL import Image
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights 
from torch.amp import autocast, GradScaler      

# ========================
# 1. Dataset and Transforms
# ========================
class MRISliceDataset(Dataset):
    """
    Custom Dataset that:
      - Reads a CSV file containing folder names and their associated label.
      - Loads a contiguous block of 10 slices (random if more than 10).
      - Applies transforms.
    """
    def __init__(self, csv_file, base_dir, transform=None):
        self.mapping = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transform

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        row = self.mapping.iloc[idx]
        folder_name = row['folder_name']
        label = int(row['label'])
        folder_path = os.path.join(self.base_dir, folder_name)
        image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        num_images = len(image_paths)
        # contiguous selection of 10
        if num_images > 10:
            start = random.randint(0, num_images - 10)
            selected = image_paths[start:start + 10]
        else:
            selected = image_paths
        seq = []
        for p in selected:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            seq.append(img)
        sequence = torch.stack(seq, dim=0)  # (10, C, H, W)
        return sequence, label

# MRI-specific normalization stats
MRI_MEAN, MRI_STD = [0.3, 0.3, 0.3], [0.15, 0.15, 0.15]

# Data augmentation and normalization for training
train_transforms = transforms.Compose([
    transforms.RandomAffine(degrees=5, translate=(0.05,0.05)),
    transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
    transforms.ToTensor(),
    transforms.Normalize(MRI_MEAN, MRI_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
])

valid_transforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MRI_MEAN, MRI_STD)
])

# ========================
# 2. Model Definition
# ========================
class SequenceClassifier(nn.Module):
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
        
        # Initialize weights for added components
        self._init_weights()
        
    def _init_weights(self):
        # Initialize weights for better training dynamics
        for m in [self.fc_proj, self.classifier]:
            if isinstance(m, nn.Sequential):
                for layer in m:
                    if isinstance(layer, nn.Linear):
                        nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
                        if layer.bias is not None:
                            nn.init.zeros_(layer.bias)

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
# 2.5 Training Pipeline with enhancements
# ========================
# Real-time plotting setup
def setup_plot():
    plt.ion()  # Turn on interactive mode
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axs[0].set_title('Loss Curves')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    
    # Accuracy plot
    axs[1].set_title('Accuracy Curves')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True)
    
    return fig, axs

def update_plot(fig, axs, epochs, train_losses, valid_losses, train_accs, valid_accs, ema_train_accs, ema_valid_accs):
    # Clear previous plots
    axs[0].clear()
    axs[1].clear()
    
    # Redraw plots with updated data
    axs[0].set_title('Loss Curves')
    axs[0].plot(epochs, train_losses, 'b-', label='Train Loss')
    axs[0].plot(epochs, valid_losses, 'r-', label='Valid Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True)
    axs[0].legend()
    
    axs[1].set_title('Accuracy Curves')
    axs[1].plot(epochs, train_accs, 'b-', label='Train Acc')
    axs[1].plot(epochs, valid_accs, 'r-', label='Valid Acc')
    axs[1].plot(epochs, ema_train_accs, 'b--', label='Train Acc (EMA)')
    axs[1].plot(epochs, ema_valid_accs, 'r--', label='Valid Acc (EMA)')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True)
    axs[1].legend()
    
    # Draw and pause to update
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)

# ========================
# 3. Training Pipeline with enhancements
# ========================
def train_model(model, train_loader, valid_loader, num_epochs=30, device='cuda'):
    model.to(device)
    # === Loss definition
    criterion = nn.CrossEntropyLoss()
    
    # === AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # === Cosine Annealing LR
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # === Mixed precision
    scaler = GradScaler()

    # === Early stopping
    best_loss, patience, wait = float('inf'), 10, 0
    
    # Tracking metrics for plotting
    epochs = []
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    # EMA tracking (alpha = 0.9 means 90% of previous value, 10% of current value)
    ema_train_acc = 0
    ema_valid_acc = 0
    ema_alpha = 0.9
    ema_train_accs = []
    ema_valid_accs = []
    
    # Setup plot
    fig, axs = setup_plot()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        batch_times = []
        
        for seqs, labs in train_loader:
            batch_start = time.time()
            seqs, labs = seqs.to(device), labs.to(device)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                outputs = model(seqs)
                loss = criterion(outputs, labs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * seqs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labs).sum().item()
            train_total += labs.size(0)
            
            batch_times.append(time.time() - batch_start)

        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Update EMA for training accuracy
        if epoch == 1:
            ema_train_acc = train_acc
        else:
            ema_train_acc = ema_alpha * ema_train_acc + (1 - ema_alpha) * train_acc

        # Validation
        model.eval()
        valid_loss, valid_correct, valid_total = 0.0, 0, 0
        with torch.no_grad():
            for seqs, labs in valid_loader:
                seqs, labs = seqs.to(device), labs.to(device)
                with autocast(device_type='cuda'):
                    outputs = model(seqs)
                    loss = criterion(outputs, labs)
                valid_loss += loss.item() * seqs.size(0)
                preds = outputs.argmax(dim=1)
                valid_correct += (preds == labs).sum().item()
                valid_total += labs.size(0)

        valid_loss /= valid_total
        valid_acc = valid_correct / valid_total
        
        # Update EMA for validation accuracy
        if epoch == 1:
            ema_valid_acc = valid_acc
        else:
            ema_valid_acc = ema_alpha * ema_valid_acc + (1 - ema_alpha) * valid_acc
        
        # Calculate metrics for logging
        avg_batch_time = sum(batch_times) / len(batch_times)
        epoch_time = time.time() - epoch_start_time
        
        # Track metrics for plotting
        epochs.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        ema_train_accs.append(ema_train_acc)
        ema_valid_accs.append(ema_valid_acc)
        
        # Update plot
        update_plot(fig, axs, epochs, train_losses, valid_losses, train_accs, valid_accs, ema_train_accs, ema_valid_accs)
        plt.savefig('training_curves.png')

        print(f"Epoch {epoch}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, EMA: {ema_train_acc:.4f} | "
              f"Valid Loss: {valid_loss:.4f}, Valid Acc: {valid_acc:.4f}, EMA: {ema_valid_acc:.4f} | "
              f"LR: {current_lr:.6f}, Time: {epoch_time:.2f}s, Batch: {avg_batch_time:.3f}s")

        # scheduler + early stop
        scheduler.step()
        if valid_loss < best_loss:
            best_loss, wait = valid_loss, 0
            torch.save(model.state_dict(), "best_sequence_classifier_CT.pth")
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    
    # Close final plot
    plt.close()

    # unfreeze backbone after warm-up if desired
    for p in model.encoder.parameters():
        p.requires_grad = True

    return model

# ========================
# 4. Putting It All Together
# ========================
def main():
    base_dir = "DATASET_DIR"  # Replace with your dataset directory
    csv_file = os.path.join(base_dir, "jpeg_mapping.csv")
    
    # Read the mapping file
    mapping_df = pd.read_csv(csv_file)
    
    # Extract subject IDs from folder names
    mapping_df['subject_id'] = mapping_df['folder_name'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    
    # Get unique subject IDs
    unique_subjects = mapping_df['subject_id'].unique()
    
    # Randomly select ~10% of subjects for the external test set
    np.random.seed(42)  # for reproducibility
    num_test_subjects = int(0.1 * len(unique_subjects))
    test_subjects = np.random.choice(unique_subjects, num_test_subjects, replace=False)
    
    # Filter the original DataFrame
    train_val_df = mapping_df[~mapping_df['subject_id'].isin(test_subjects)].copy()
    test_df = mapping_df[mapping_df['subject_id'].isin(test_subjects)].copy()
    
    # Save filtered CSVs temporarily
    train_val_csv = "train_val_mapping.csv"
    test_csv = "test_mapping.csv"
    train_val_df.to_csv(train_val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # Create datasets
    train_val_dataset = MRISliceDataset(train_val_csv, base_dir, transform=train_transforms)
    test_dataset = MRISliceDataset(test_csv, base_dir, transform=valid_transforms)
    
    # Split train_val into train and validation
    train_size = int(0.8 * len(train_val_dataset))
    valid_size = len(train_val_dataset) - train_size
    train_ds, valid_ds = random_split(train_val_dataset, [train_size, valid_size])
    valid_ds.dataset.transform = valid_transforms
    
    # Create data loaders
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_ds, batch_size=8, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceClassifier(feature_dim=128, hidden_dim=64, num_layers=1, num_classes=6)
    
    model = torch.compile(
        model,
        options={
            "epilogue_fusion": True,    # Fuses pointwise operations
            "triton.cudagraphs": True   # Reduces Python-CUDA overhead
        }
    )
    print("Model compiled.")
    
    trained = train_model(model, train_loader, valid_loader, num_epochs=30, device=device)
    print("Training complete.")
    
    # Evaluate on external test set
    model.load_state_dict(torch.load("best_sequence_classifier_for.pth"))
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for seqs, labs in test_loader:
            seqs, labs = seqs.to(device), labs.to(device)
            with autocast(device_type='cuda'):
                outputs = model(seqs)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labs).sum().item()
            test_total += labs.size(0)
    test_acc = test_correct / test_total
    print(f"External Test Set Accuracy: {test_acc:.4f}")
    
    # Clean up temporary files
    os.remove(train_val_csv)
    os.remove(test_csv)

if __name__ == "__main__":
    main()