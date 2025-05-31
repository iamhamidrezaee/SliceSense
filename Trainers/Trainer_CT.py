# Trainer_MRI_Enhanced.py
# Copyright (c) 2025 — SliceSense
# --------------------------------------------------------------------------------------
# Train a ResNet-18 ➜ BiLSTM ➜ Self-Attention model to classify MRI slice-thickness
# from folders of 2-D JPEG slices (10-slice contiguous blocks).
# Enhanced version with performance optimizations and consistency with CT trainer
# --------------------------------------------------------------------------------------
# Usage (single-GPU):
#     python Trainer_MRI_Enhanced.py
# --------------------------------------------------------------------------------------
import os
import glob
import random
import time
import warnings
from typing import Sequence
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision.models import resnet18, ResNet18_Weights
from torch.amp import autocast, GradScaler

# Silence dynamo warnings coming from torch.compile
import torch._dynamo
torch._dynamo.config.suppress_errors = True
warnings.filterwarnings("ignore", category=UserWarning)

# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed()

# ======================================================================================
# 1. Dataset and Transforms
# ======================================================================================
class MRISliceDataset(Dataset):
    """
    Custom Dataset that:
      - Reads a CSV file containing folder names and their associated label.
      - Loads a contiguous block of 10 slices (deterministic for val/test).
      - Applies transforms.
    """
    def __init__(
        self,
        csv_file: str,
        base_dir: str,
        transform: transforms.Compose | None = None,
        is_train: bool = True,  # Flag to determine slice selection behavior
    ):
        self.mapping = pd.read_csv(csv_file)
        self.base_dir = base_dir
        self.transform = transform
        self.is_train = is_train
        
        # Create a seed for each folder (used for deterministic validation)
        self.folder_seeds = {row["folder_name"]: hash(row["folder_name"]) % 10000 
                             for _, row in self.mapping.iterrows()}
    
    def __len__(self) -> int:
        return len(self.mapping)
    
    def __getitem__(self, idx: int):
        row = self.mapping.iloc[idx]
        folder_name = row['folder_name']
        label = int(row['label'])
        folder_path = os.path.join(self.base_dir, folder_name)
        image_paths: Sequence[str] = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        
        if len(image_paths) == 0:
            raise RuntimeError(f"No JPEG slices found in {folder_path}")
        
        num_images = len(image_paths)
        
        if self.is_train:
            # In training mode, randomly select slices each time
            if num_images >= 10:
                start = random.randint(0, num_images - 10)
                selected = image_paths[start:start + 10]
            else:
                # Pad with last image if fewer than 10
                selected = image_paths + [image_paths[-1]] * (10 - num_images)
        else:
            # In validation/test mode, deterministically select slices
            folder_seed = self.folder_seeds[row["folder_name"]]
            rand_gen = random.Random(folder_seed)
            
            if num_images >= 10:
                # Always pick the same slice window for this folder
                start = rand_gen.randint(0, num_images - 10)
                selected = image_paths[start:start + 10]
            else:
                selected = image_paths + [image_paths[-1]] * (10 - num_images)
        
        seq: list[torch.Tensor] = []
        for p in selected:
            img = Image.open(p).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            seq.append(img)
        
        sequence = torch.stack(seq, dim=0)  # (10, C, H, W)
        return sequence, label

# --------------------------------------------------------------------------------------
# MRI-specific transforms (realistic for clinical scenarios)
# --------------------------------------------------------------------------------------
# MRI-specific normalization stats
MRI_MEAN: list[float] = [0.3, 0.3, 0.3]
MRI_STD: list[float] = [0.15, 0.15, 0.15]

# Training transforms
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)), 
    transforms.RandomHorizontalFlip(p=0.5), 
    transforms.RandomRotation(5),  
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.03, 0.03),
        scale=(0.98, 1.02)  
    ),
    transforms.ColorJitter(
        brightness=0.1,  
        contrast=0.1,    
        saturation=0,    
        hue=0          
    ),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)), 
    transforms.ToTensor(),
    transforms.Normalize(MRI_MEAN, MRI_STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.05), ratio=(0.3, 3.3)) 
])

# Validation transforms
valid_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(MRI_MEAN, MRI_STD),
])

# ======================================================================================
# 2. Model Definition – ResNet encoder + Positional Emb. + BiLSTM + Multi-Head Attention
# ======================================================================================
class SequenceClassifier(nn.Module):
    """
    Enhanced architecture with positional embeddings for better sequence understanding.
    
    Args:
        feature_dim : dim after CNN encoder (before LSTM)
        hidden_dim  : LSTM hidden size
        num_layers  : # BiLSTM layers
        num_classes : 6 (slice-thickness classes)
        max_slices  : fixed length of MRI sequences (10)
    """
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        max_slices: int = 10,
    ):
        super().__init__()
        
        # ---------- Per-slice CNN encoder -----------------------------------
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()  # Remove final FC, output 512-D
        self.encoder = backbone
        
        # Project to smaller feature_dim & add GELU non-linearity
        self.fc_proj = nn.Sequential(
            nn.Linear(512, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        
        # Learnable positional embedding (crucial for slice ordering)
        self.max_slices = max_slices
        self.pos_embed = nn.Parameter(torch.zeros(max_slices, feature_dim))
        
        # ---------- Sequence modeller: BiLSTM + MH-Attention ---------------
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0,
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # *2 for bidirectional
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        
        # ---------- Classification MLP --------------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Initialize weights for better training dynamics
        for m in [self.fc_proj, self.classifier]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    
    def forward(self, x):  # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        assert S <= self.max_slices, f"Sequence length {S} exceeds max_slices={self.max_slices}"
        
        # Extract features from each image in sequence
        x = x.view(B * S, C, H, W)           # (B*S, C, H, W)
        feats = self.encoder(x)               # (B*S, 512)
        feats = self.fc_proj(feats)           # (B*S, feature_dim)
        feats = feats.view(B, S, -1)          # (B, S, feature_dim)
        
        # Add positional embedding
        feats = feats + self.pos_embed[:S].unsqueeze(0)
        
        # Process sequence with BiLSTM
        lstm_out, _ = self.lstm(feats)       # (B, S, 2*hidden_dim)
        
        # Apply self-attention to capture relationships between slices
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global pooling (weighted combination of avg and max)
        avg_pool = attn_out.mean(dim=1)
        max_pool, _ = attn_out.max(dim=1)
        pooled = 0.5 * avg_pool + 0.5 * max_pool
        
        # Classification
        return self.classifier(pooled)

# ======================================================================================
# 2.5 Plotting helpers (real-time loss/acc curves)
# ======================================================================================
def setup_plot():
    plt.ion()  # Turn on interactive mode
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axs[0].set_title('Loss Curves')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Cross-Entropy Loss')
    axs[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axs[1].set_title('Accuracy Curves')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].grid(True, alpha=0.3)
    
    return fig, axs

def update_plot(
    fig,
    axs,
    epochs,
    train_losses,
    valid_losses,
    train_accs,
    valid_accs,
    ema_train_accs,
    ema_valid_accs,
):
    # Clear previous plots
    axs[0].clear()
    axs[1].clear()
    
    # Loss curves
    axs[0].plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    axs[0].plot(epochs, valid_losses, 'r-', label='Valid Loss', linewidth=2)
    axs[0].set_title('Loss Curves')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Cross-Entropy Loss')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    axs[1].plot(epochs, train_accs, 'b-', label='Train Acc', linewidth=2)
    axs[1].plot(epochs, valid_accs, 'r-', label='Valid Acc', linewidth=2)
    axs[1].plot(epochs, ema_train_accs, 'b--', label='Train Acc (EMA)', alpha=0.7)
    axs[1].plot(epochs, ema_valid_accs, 'r--', label='Valid Acc (EMA)', alpha=0.7)
    axs[1].set_title('Accuracy Curves')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Draw and pause to update
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)
    
    # Save figure
    plt.savefig('training_curves_MRI.png', dpi=150, bbox_inches='tight')

# ======================================================================================
# 3. Training routine with mixup and label smoothing
# ======================================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int = 80,
    device: str = "cuda",
):
    model.to(device)
    
    # Loss with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # AdamW optimizer
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # Cosine Annealing LR scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # Mixed precision training
    scaler = GradScaler()
    
    # Early stopping parameters
    best_acc = 0.0
    patience = 15
    wait = 0
    
    # Tracking metrics for plotting
    epochs = []
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    
    # EMA tracking
    ema_train_acc = 0
    ema_valid_acc = 0
    ema_alpha = 0.9
    ema_train_accs = []
    ema_valid_accs = []
    
    # Setup plot
    fig, axs = setup_plot()
    
    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        # ---------------- TRAIN ------------------------------------------
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        batch_times = []
        
        for batch_idx, (seqs, labs) in enumerate(train_loader):
            batch_start = time.time()
            seqs, labs = seqs.to(device), labs.to(device)
            
            # Mixup augmentation after epoch 5
            if epoch > 5 and random.random() < 0.5:  # Apply mixup 50% of the time
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(seqs.size(0), device=device)
                seqs_mixed = lam * seqs + (1 - lam) * seqs[idx]
                labels_a, labels_b = labs, labs[idx]
                mixup = True
            else:
                seqs_mixed = seqs
                mixup = False
            
            optimizer.zero_grad(set_to_none=True)
            
            with autocast(device_type='cuda'):
                outputs = model(seqs_mixed)
                if mixup:
                    loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
                else:
                    loss = criterion(outputs, labs)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item() * labs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labs).sum().item()
            train_total += labs.size(0)
            
            batch_times.append(time.time() - batch_start)
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Update EMA for training accuracy
        if epoch == 1:
            ema_train_acc = train_acc
        else:
            ema_train_acc = ema_alpha * ema_train_acc + (1 - ema_alpha) * train_acc
        
        # ---------------- VALIDATION ------------------------------------------
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        valid_total = 0
        
        with torch.no_grad():
            for seqs, labs in valid_loader:
                seqs, labs = seqs.to(device), labs.to(device)
                
                with autocast(device_type='cuda'):
                    outputs = model(seqs)
                    loss = criterion(outputs, labs)
                
                valid_loss += loss.item() * labs.size(0)
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
        
        # ---------------- Scheduler / Logging ----------------------------
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()
        
        # Calculate timing metrics
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
        update_plot(fig, axs, epochs, train_losses, valid_losses, 
                   train_accs, valid_accs, ema_train_accs, ema_valid_accs)
        
        print(f"Epoch {epoch:03d}/{num_epochs} | "
              f"Loss: {train_loss:.4f}/{valid_loss:.4f} | "
              f"Acc: {train_acc:.4f}/{valid_acc:.4f} (EMA: {ema_valid_acc:.4f}) | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s (batch: {avg_batch_time:.3f}s)")
        
        # ---------------- Checkpointing & Early Stopping -------------------
        if valid_acc > best_acc:
            best_acc = valid_acc
            wait = 0
            torch.save(model.state_dict(), "best_sequence_classifier_MRI.pth")
            print(f"  → New best model saved! (Valid Acc: {valid_acc:.4f})")
        else:
            wait += 1
            if wait >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
    
    plt.close()
    print(f"\nTraining complete! Best validation accuracy: {best_acc:.4f}")
    
    return model

# ======================================================================================
# 4. Test-Time Augmentation (TTA) Functions
# ======================================================================================
def five_crop_transform(img):
    """Apply five crop transform to an image."""
    five_crop = transforms.FiveCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(MRI_MEAN, MRI_STD)
    
    crops = five_crop(img)
    result = []
    for crop in crops:
        tensor = to_tensor(crop)
        normalized = normalize(tensor)
        result.append(normalized)
    return result

def get_two_window_positions(image_paths, num_images, window_size=10):
    """Get start positions for two windows at approximately 1/3 and 2/3 of the volume."""
    if num_images <= window_size:
        # If we don't have enough slices, just return one window
        return [0]
    
    # Calculate positions at 1/3 and 2/3 of the volume
    one_third_pos = max(0, int(num_images * 1/3) - window_size//2)
    two_third_pos = max(0, min(int(num_images * 2/3) - window_size//2, num_images - window_size))
    
    # If windows overlap too much, use just one window
    if abs(one_third_pos - two_third_pos) < 5:
        return [max(0, int(num_images/2) - window_size//2)]
    
    return [one_third_pos, two_third_pos]

def evaluate_with_tta(model, dataset, base_dir, device, batch_size=8):
    """Evaluate model using test-time augmentation with 5-crop and two slice windows."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("Performing test-time augmentation evaluation...")
    
    for idx in range(len(dataset.mapping)):
        if idx % 100 == 0:
            print(f"  Processing sample {idx}/{len(dataset.mapping)}...")
        
        row = dataset.mapping.iloc[idx]
        folder_path = os.path.join(base_dir, row["folder_name"])
        label = int(row["label"])
        image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        
        if len(image_paths) == 0:
            raise RuntimeError(f"No JPEG slices found in {folder_path}")
        
        num_images = len(image_paths)
        
        # Get start positions for two windows
        window_positions = get_two_window_positions(image_paths, num_images, window_size=10)
        
        all_logits = []
        
        for start_pos in window_positions:
            # Select window of 10 slices
            if num_images >= 10:
                selected = image_paths[start_pos:start_pos + 10]
            else:
                selected = image_paths + [image_paths[-1]] * (10 - num_images)
            
            # Process 5 crops for each image in the sequence
            all_crops_seq = []
            for p in selected:
                img = Image.open(p).convert("RGB")
                img = img.resize((256, 256))
                crops = five_crop_transform(img)
                all_crops_seq.append(crops)
            
            # Reorganize crops into 5 sequences
            sequences = []
            for crop_idx in range(5):  # For each of the 5 spatial crops
                sequence = []
                for slice_idx in range(10):  # For each of the 10 slices
                    sequence.append(all_crops_seq[slice_idx][crop_idx])
                sequences.append(torch.stack(sequence, dim=0))  # (10, C, H, W)
            
            # Process in batches to avoid memory issues
            crop_logits = []
            for i in range(0, len(sequences), batch_size):
                batch_seqs = sequences[i:i+batch_size]
                if batch_seqs:
                    batch_tensor = torch.stack(batch_seqs, dim=0).to(device)
                    with torch.no_grad(), autocast(device_type="cuda"):
                        outputs = model(batch_tensor)
                    crop_logits.append(outputs.cpu())
            
            # Combine crop logits for this window
            if crop_logits:
                window_logits = torch.cat(crop_logits, dim=0).mean(dim=0, keepdim=True)
                all_logits.append(window_logits)
        
        # Average logits across windows
        final_logits = torch.cat(all_logits, dim=0).mean(dim=0)
        pred = final_logits.argmax().item()
        
        all_preds.append(pred)
        all_labels.append(label)
    
    # Calculate accuracy
    correct = sum(p == l for p, l in zip(all_preds, all_labels))
    accuracy = correct / len(all_labels)
    
    # Calculate per-class metrics
    classes = range(6)
    class_metrics = {}
    
    for cls in classes:
        cls_indices = [i for i, label in enumerate(all_labels) if label == cls]
        if cls_indices:
            cls_correct = sum(all_preds[i] == all_labels[i] for i in cls_indices)
            cls_acc = cls_correct / len(cls_indices)
            class_metrics[cls] = {'accuracy': cls_acc, 'count': len(cls_indices)}
        else:
            class_metrics[cls] = {'accuracy': 0.0, 'count': 0}
    
    return accuracy, all_preds, all_labels, class_metrics

# ======================================================================================
# 5. Stratified Split Helper Functions
# ======================================================================================
def stratified_subject_split(mapping_df, test_size=0.1, random_state=42):
    """
    Perform stratified split by subject while maintaining label distribution.
    """
    # Create subject-level dataframe with majority label per subject
    subject_labels = mapping_df.groupby('subject_id')['label'].agg(lambda x: x.mode()[0]).reset_index()
    
    # Perform stratified split on subjects
    train_val_subjects, test_subjects = train_test_split(
        subject_labels['subject_id'],
        test_size=test_size,
        stratify=subject_labels['label'],
        random_state=random_state
    )
    
    # Filter original dataframe
    train_val_df = mapping_df[mapping_df['subject_id'].isin(train_val_subjects)].copy()
    test_df = mapping_df[mapping_df['subject_id'].isin(test_subjects)].copy()
    
    return train_val_df, test_df

def create_stratified_dataset_split(dataset, train_ratio=0.8, random_state=42):
    """
    Create stratified train/validation split from a dataset.
    """
    # Get all labels from the dataset
    labels = []
    for i in range(len(dataset)):
        _, label = dataset[i]
        labels.append(label)
    
    # Create stratified indices
    indices = np.arange(len(dataset))
    train_indices, val_indices = train_test_split(
        indices,
        train_size=train_ratio,
        stratify=labels,
        random_state=random_state
    )
    
    # Create subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    
    return train_subset, val_subset

def print_distribution(name, labels):
    """Print label distribution for a dataset split."""
    counter = Counter(labels)
    total = sum(counter.values())
    print(f"\n{name} Distribution:")
    for label in sorted(counter.keys()):
        count = counter[label]
        percentage = (count / total) * 100
        print(f"  Label {label}: {count} ({percentage:.2f}%)")

# ======================================================================================
# 6. Main Training and Evaluation Pipeline
# ======================================================================================
def main():
    # Set random seeds
    set_seed(42)
    
    # Configuration
    base_dir = "/home/hr328/SliceSense/SliceSense_Unified_Dataset"
    csv_file = os.path.join(base_dir, "unified_mapping.csv")
    
    # Read the mapping file
    mapping_df = pd.read_csv(csv_file)
    
    # Extract subject IDs from folder names
    mapping_df['subject_id'] = mapping_df['folder_name'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    
    # Print original distribution
    print_distribution("Original Dataset", mapping_df['label'].tolist())
    
    # Perform stratified split by subject (10% for test)
    train_val_df, test_df = stratified_subject_split(mapping_df, test_size=0.1, random_state=42)
    
    # Print distributions after subject-level split
    print_distribution("Train+Val Set", train_val_df['label'].tolist())
    print_distribution("Test Set", test_df['label'].tolist())
    
    # Save filtered CSVs temporarily
    train_val_csv = "train_val_mapping_mri.csv"
    test_csv = "test_mapping_mri.csv"
    train_val_df.to_csv(train_val_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # Create datasets with proper transforms
    print("\nCreating datasets...")
    train_val_dataset = MRISliceDataset(train_val_csv, base_dir, transform=train_transforms, is_train=True)
    test_dataset = MRISliceDataset(test_csv, base_dir, transform=valid_transforms, is_train=False)
    
    # Perform stratified train/validation split (80/20)
    print("Creating train/validation split...")
    train_ds, valid_ds = create_stratified_dataset_split(train_val_dataset, train_ratio=0.8, random_state=42)
    
    # Create separate dataset for validation with validation transforms
    valid_dataset = MRISliceDataset(train_val_csv, base_dir, transform=valid_transforms, is_train=False)
    valid_ds = Subset(valid_dataset, valid_ds.indices)
    
    # Get labels for verification and balanced sampling
    train_labels = []
    valid_labels = []
    
    print("Verifying stratification...")
    for i in range(len(train_ds)):
        _, label = train_ds[i]
        train_labels.append(label)
    
    for i in range(len(valid_ds)):
        _, label = valid_ds[i]
        valid_labels.append(label)
    
    print_distribution("Training Set", train_labels)
    print_distribution("Validation Set", valid_labels)
    
    # Create balanced sampler for training
    print("\nCreating balanced sampler for training...")
    class_counts = Counter(train_labels)
    class_weights = 1.0 / torch.tensor([class_counts[i] for i in range(6)], dtype=torch.float)
    sample_weights = torch.tensor([class_weights[label] for label in train_labels])
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader = DataLoader(
        train_ds, 
        batch_size=32,  
        sampler=sampler, 
        num_workers=10, 
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=True
    )
    
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True
    )
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    model = SequenceClassifier(
        feature_dim=256, 
        hidden_dim=128, 
        num_layers=2, 
        num_classes=6,
        max_slices=10
    )
    
    # Compile model for better performance
    try:
        model = torch.compile(model)
        print("Model compiled successfully.")
    except:
        print("Model compilation failed, using uncompiled model.")
    
    # Train model
    print("\nStarting training...")
    trained_model = train_model(model, train_loader, valid_loader, num_epochs=80, device=device)
    
    # Evaluate on external test set (without TTA)
    print("\nEvaluating on test set (without TTA)...")
    model.load_state_dict(torch.load("best_sequence_classifier_MRI.pth", map_location=device))
    model.eval()
    
    test_correct = 0
    test_total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for seqs, labs in test_loader:
            seqs, labs = seqs.to(device), labs.to(device)
            with autocast(device_type='cuda'):
                outputs = model(seqs)
            preds = outputs.argmax(dim=1)
            test_correct += (preds == labs).sum().item()
            test_total += labs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    
    test_acc = test_correct / test_total
    print(f"\nTest Set Accuracy (without TTA): {test_acc:.4f}")
    
    # Print per-class accuracy (without TTA)
    print("\nPer-class Test Accuracy (without TTA):")
    for label in range(6):
        mask = np.array(all_labels) == label
        if mask.sum() > 0:
            class_acc = (np.array(all_preds)[mask] == label).mean()
            print(f"  Class {label}: {class_acc:.4f} ({mask.sum()} samples)")
    
    # Test-time augmentation evaluation
    print("\n" + "="*80)
    print("Evaluating with test-time augmentation (5-crop + 2-window)...")
    tta_acc, tta_preds, tta_labels, tta_class_metrics = evaluate_with_tta(
        model, test_dataset, base_dir, device
    )
    
    print(f"\nTest Accuracy with TTA: {tta_acc:.4f}")
    print(f"Improvement from TTA: +{(tta_acc - test_acc)*100:.2f}%")
    
    # Print per-class accuracy with TTA
    print("\nPer-class Test Accuracy with TTA:")
    for cls in range(6):
        metrics = tta_class_metrics[cls]
        print(f"  Class {cls}: {metrics['accuracy']:.4f} ({metrics['count']} samples)")
    
    # Clean up temporary files
    try:
        os.remove(train_val_csv)
        os.remove(test_csv)
        print("\nTemporary files cleaned up.")
    except:
        print("\nNote: Could not remove temporary CSV files.")
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()