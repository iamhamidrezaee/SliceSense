# Trainer_CT.py
# Copyright (c) 2025 — SliceSense
# --------------------------------------------------------------------------------------
# Train a ResNet-18 ➜ BiLSTM ➜ Self-Attention model to classify abdominal-CT slice-thickness
# from folders of 2-D JPEG slices (25-slice contiguous blocks).
# --------------------------------------------------------------------------------------
# Usage (single-GPU):
#     python Trainer_CT.py
# --------------------------------------------------------------------------------------
import os
import glob
import random
import time
import warnings
from typing import Sequence
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
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
class CTSliceDataset(Dataset):
    """
    Each row of `csv_file` lists a folder of JPEG slices + a label (0-5).
    **getitem** returns:
        sequence (torch.FloatTensor): shape (S=25, C, H, W)
        label    (int):               0‒5
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
        folder_path = os.path.join(self.base_dir, row["folder_name"])
        label = int(row["label"])
        image_paths: Sequence[str] = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        if len(image_paths) == 0:
            raise RuntimeError(f"No JPEG slices found in {folder_path}")
        # Pick a contiguous window of 25 slices
        num_images = len(image_paths)
        
        if self.is_train:
            # In training mode, randomly select slices each time
            if num_images >= 25:
                start = random.randint(0, num_images - 25)
                selected = image_paths[start : start + 25]
            else:
                selected = image_paths + [image_paths[-1]] * (25 - num_images)
        else:
            # In validation/test mode, deterministically select slices
            folder_seed = self.folder_seeds[row["folder_name"]]
            rand_gen = random.Random(folder_seed)
            
            if num_images >= 25:
                # Always pick the same slice window for this folder
                start = rand_gen.randint(0, num_images - 25)
                selected = image_paths[start : start + 25]
            else:
                selected = image_paths + [image_paths[-1]] * (25 - num_images)
        seq: list[torch.Tensor] = []
        for p in selected:
            img = Image.open(p).convert("RGB")
            if self.transform is not None:
                img = self.transform(img)
            seq.append(img)
        sequence = torch.stack(seq, dim=0)  # (25, C, H, W)
        return sequence, label
# --------------------------------------------------------------------------------------
# CT-specific transforms
# --------------------------------------------------------------------------------------
CT_MEAN: list[float] = [0.242, 0.242, 0.242]
CT_STD: list[float] = [0.121, 0.121, 0.121]

train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05)), 
    transforms.ColorJitter(brightness=0.05, contrast=0.05),
    transforms.ToTensor(),
    transforms.Normalize(CT_MEAN, CT_STD),
])
valid_transforms = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(CT_MEAN, CT_STD),
    ]
)
# ======================================================================================
# 2. Model Definition – ResNet encoder + Positional Emb. + BiLSTM + Multi-Head Attention
# ======================================================================================
class SequenceClassifier(nn.Module):
    """
    Args
    ----
    feature_dim : dim after CNN encoder (before LSTM)
    hidden_dim  : LSTM hidden size
    num_layers  : # BiLSTM layers
    num_classes : 6 (slice-thickness classes)
    max_slices  : fixed length of CT sequences (25)
    """
    def __init__(
        self,
        feature_dim: int = 256,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_classes: int = 6,
        max_slices: int = 25,
    ):
        super().__init__()
        # ---------- Per-slice CNN encoder -----------------------------------
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()  # output 512-D
        self.encoder = backbone
        # Project to smaller feature_dim & add GELU non-linearity
        self.fc_proj = nn.Sequential(
            nn.Linear(512, feature_dim, bias=False),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )
        # Learnable positional embedding (helps model distinguish slice order)
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
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
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
    # ----------------------------------------------------------------------
    def _init_weights(self):
        # Kaiming init for newly added Linear layers
        for m in [self.fc_proj, self.classifier]:
            for layer in m:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="leaky_relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    # ----------------------------------------------------------------------
    def forward(self, x):  # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        assert (
            S <= self.max_slices
        ), f"Sequence length {S} exceeds max_slices={self.max_slices}"
        # CNN features ------------------------------------------------------
        x = x.view(B * S, C, H, W)           # (B*S, C, H, W)
        feats = self.encoder(x)              # (B*S, 512)
        feats = self.fc_proj(feats)          # (B*S, feature_dim)
        feats = feats.view(B, S, -1)         # (B, S, feature_dim)
        # Add positional embedding
        feats = feats + self.pos_embed[:S].unsqueeze(0)
        # BiLSTM ------------------------------------------------------------
        lstm_out, _ = self.lstm(feats)       # (B, S, 2*hidden*dim)
        # Self-Attention ----------------------------------------------------
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        # Pooling (avg + max) ----------------------------------------------
        avg_pool = attn_out.mean(dim=1)
        max_pool, _ = attn_out.max(dim=1)
        pooled = 0.5 * avg_pool + 0.5 * max_pool
        return self.classifier(pooled)
# ======================================================================================
# 2.5 Plotting helpers (real-time loss/acc curves)
# ======================================================================================
def setup_plot():
    plt.ion()
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    axs[0].set_title("Loss")
    axs[1].set_title("Accuracy")
    for ax in axs:
        ax.set_xlabel("Epoch")
        ax.grid(True)
    axs[0].set_ylabel("Cross-Entropy")
    axs[1].set_ylabel("Acc")
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
    axs[0].cla()
    axs[1].cla()
    axs[0].plot(epochs, train_losses, "b-", label="train")
    axs[0].plot(epochs, valid_losses, "r-", label="valid")
    axs[0].set_title("Loss")
    axs[0].legend()
    axs[0].grid(True)
    axs[1].plot(epochs, train_accs, "b-", label="train")
    axs[1].plot(epochs, valid_accs, "r-", label="valid")
    axs[1].plot(epochs, ema_train_accs, "b--", label="train EMA")
    axs[1].plot(epochs, ema_valid_accs, "r--", label="valid EMA")
    axs[1].set_title("Accuracy")
    axs[1].legend()
    axs[1].grid(True)
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.1)
    
    # Save figure after each update
    plt.savefig("training_curves_CT.png")
# ======================================================================================
# 3. Training routine
# ======================================================================================
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    num_epochs: int = 180,
    device: str = "cuda",
):
    model.to(device)
    
    # --- LOSS -------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # --- OPTIMISER --------------------------------------------------------
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    
    # --- LR SCHEDULER (cosine LR) ------------------------------
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    
    # --- MIXED PRECISION --------------------------------------------------
    scaler = GradScaler()
    # Tracking
    best_acc = 0.0
    epochs, train_losses, valid_losses = [], [], []
    train_accs, valid_accs, ema_train_accs, ema_valid_accs = [], [], [], []
    ema_train_acc = ema_valid_acc = 0.0
    ema_alpha = 0.9
    fig, axs = setup_plot()
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        # ---------------- TRAIN ------------------------------------------
        model.train()
        train_loss = train_correct = train_total = 0
        batch_times = []
        for seqs, labs in train_loader:
            batch_start = time.time()
            seqs, labs = seqs.to(device), labs.to(device)
            # mixup after epoch 5
            mixup = epoch > 5
            if mixup:
                lam = np.random.beta(0.2, 0.2)
                idx = torch.randperm(seqs.size(0), device=device)
                seqs_mixed = lam * seqs + (1 - lam) * seqs[idx]
                labels_a, labels_b = labs, labs[idx]
            else:
                seqs_mixed = seqs
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda"):
                outputs = model(seqs_mixed)
                if mixup:
                    loss = (
                        lam * criterion(outputs, labels_a)
                        + (1 - lam) * criterion(outputs, labels_b)
                    )
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
        ema_train_acc = (
            train_acc if epoch == 1 else ema_alpha * ema_train_acc + (1 - ema_alpha) * train_acc
        )
        # ---------------- VALID ------------------------------------------
        model.eval()
        valid_loss = valid_correct = valid_total = 0
        with torch.no_grad():
            for seqs, labs in valid_loader:
                seqs, labs = seqs.to(device), labs.to(device)
                with autocast(device_type="cuda"):
                    outputs = model(seqs)
                    loss = criterion(outputs, labs)
                valid_loss += loss.item() * labs.size(0)
                valid_correct += (outputs.argmax(1) == labs).sum().item()
                valid_total += labs.size(0)
        valid_loss /= valid_total
        valid_acc = valid_correct / valid_total
        ema_valid_acc = (
            valid_acc if epoch == 1 else ema_alpha * ema_valid_acc + (1 - ema_alpha) * valid_acc
        )
        # ---------------- Scheduler / Logging ----------------------------
        scheduler.step(epoch + 1)  # SGDR expects "epoch + 1"
        lrs = ", ".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)
        epochs.append(epoch)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        train_accs.append(train_acc)
        valid_accs.append(valid_acc)
        ema_train_accs.append(ema_train_acc)
        ema_valid_accs.append(ema_valid_acc)
        update_plot(
            fig,
            axs,
            epochs,
            train_losses,
            valid_losses,
            train_accs,
            valid_accs,
            ema_train_accs,
            ema_valid_accs,
        )
        epoch_time = time.time() - epoch_start
        avg_bt = sum(batch_times) / len(batch_times)
        print(
            f"Epoch {epoch:03d}/{num_epochs} | "
            f"loss {train_loss:.4f}/{valid_loss:.4f} | "
            f"acc {train_acc:.3f}/{valid_acc:.3f} (EMA {ema_valid_acc:.3f}) | "
            f"LR {lrs} | {epoch_time:.1f}s (batch {avg_bt:.2f}s)"
        )
        # ---------------- Checkpoint -------------------------------------
        if valid_acc > best_acc:
            best_acc = valid_acc
            torch.save(model.state_dict(), "best_sequence_classifier_CT.pth")
    plt.close()
    print(f"Training complete — best valid accuracy {best_acc:.3f}")
    return model

# Test-time augmentation functions
def five_crop_transform(img):
    """Apply five crop transform to an image."""
    five_crop = transforms.FiveCrop(224)
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(CT_MEAN, CT_STD)
    
    crops = five_crop(img)
    result = []
    for crop in crops:
        tensor = to_tensor(crop)
        normalized = normalize(tensor)
        result.append(normalized)
    return result

def get_two_window_positions(image_paths, num_images):
    """Get start positions for two windows at approximately 1/3 and 2/3 of the volume."""
    if num_images <= 25:
        # If we don't have enough slices, just return one window
        return [0]
    
    # Calculate positions at 1/3 and 2/3 of the volume
    one_third_pos = max(0, int(num_images * 1/3) - 12)  # center window at 1/3
    two_third_pos = max(0, min(int(num_images * 2/3) - 12, num_images - 25))  # center at 2/3
    
    # If windows overlap too much, use just one window
    if abs(one_third_pos - two_third_pos) < 10:
        return [max(0, int(num_images/2) - 12)]  # use middle window
    
    return [one_third_pos, two_third_pos]

def evaluate_with_tta(model, dataset, base_dir, device, batch_size=8):
    """Evaluate model using test-time augmentation with 5-crop and two slice windows."""
    model.eval()
    all_preds = []
    all_labels = []
    
    for idx in range(len(dataset.mapping)):
        row = dataset.mapping.iloc[idx]
        folder_path = os.path.join(base_dir, row["folder_name"])
        label = int(row["label"])
        image_paths = sorted(glob.glob(os.path.join(folder_path, "*.jpg")))
        
        if len(image_paths) == 0:
            raise RuntimeError(f"No JPEG slices found in {folder_path}")
        
        num_images = len(image_paths)
        
        # Get start positions for two windows
        window_positions = get_two_window_positions(image_paths, num_images)
        
        all_logits = []
        
        for start_pos in window_positions:
            # Select window of 25 slices
            if num_images >= 25:
                selected = image_paths[start_pos:start_pos + 25]
            else:
                selected = image_paths + [image_paths[-1]] * (25 - num_images)
            
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
                for slice_idx in range(25):  # For each of the 25 slices
                    sequence.append(all_crops_seq[slice_idx][crop_idx])
                sequences.append(torch.stack(sequence, dim=0))  # (25, C, H, W)
            
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
    classes = range(6)  # Assuming 6 classes (0-5)
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
# 4. Putting It All Together
# ======================================================================================
def main():
    # Data directory setup
    base_dir = "DATASET_PATH"  # Replace with your dataset path
    csv_file = os.path.join(base_dir, "jpeg_mapping.csv")
    # Load the mapping data
    mapping_df = pd.read_csv(csv_file)
    
    # Extract subject IDs from folder names
    mapping_df["subject_id"] = mapping_df["folder_name"].apply(lambda x: "_".join(x.split("_")[:-1]))
    # Get unique subjects
    unique_subjects = mapping_df["subject_id"].unique()
    
    # Step 1: Split subjects into test (10%) and train+val (90%)
    test_subjects, train_val_subjects = train_test_split(
        unique_subjects, test_size=0.9, random_state=42
    )
    
    # Step 2: Further split train+val subjects into train (80% of 90%) and validation (20% of 90%)
    train_subjects, valid_subjects = train_test_split(
        train_val_subjects, test_size=0.2, random_state=42
    )
    
    # Create dataframes for each split based on subject IDs
    train_df = mapping_df[mapping_df["subject_id"].isin(train_subjects)].copy()
    valid_df = mapping_df[mapping_df["subject_id"].isin(valid_subjects)].copy()
    test_df = mapping_df[mapping_df["subject_id"].isin(test_subjects)].copy()
    
    # Save split mappings to CSV files
    train_csv = "train_mapping.csv"
    valid_csv = "valid_mapping.csv"
    test_csv = "test_mapping.csv"
    
    train_df.to_csv(train_csv, index=False)
    valid_df.to_csv(valid_csv, index=False)
    test_df.to_csv(test_csv, index=False)
    
    # Log split sizes for verification
    print(f"Total subjects: {len(unique_subjects)}")
    print(f"Total examples: {len(mapping_df)}")
    print(f"Train split: {len(train_subjects)} subjects, {len(train_df)} examples")
    print(f"Valid split: {len(valid_subjects)} subjects, {len(valid_df)} examples")
    print(f"Test split: {len(test_subjects)} subjects, {len(test_df)} examples")
    
    # Create datasets with appropriate transforms and deterministic behavior for val/test
    train_dataset = CTSliceDataset(train_csv, base_dir, transform=train_transforms, is_train=True)
    valid_dataset = CTSliceDataset(valid_csv, base_dir, transform=valid_transforms, is_train=False)
    test_dataset = CTSliceDataset(test_csv, base_dir, transform=valid_transforms, is_train=False)
    
    # Balanced class-aware sampler for training
    class_counts = train_df['label'].value_counts().sort_index()
    class_weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float)
    sample_weights = class_weights[train_df['label'].values]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,            
        sampler=sampler,  # Using our balanced sampler instead of shuffle=True
        num_workers=10,           
        pin_memory=True,
        persistent_workers=True,  
        prefetch_factor=4,       
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=32, shuffle=False, num_workers=10, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=32, shuffle=False, num_workers=10, pin_memory=True
    )
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Create and compile model
    model = SequenceClassifier(feature_dim=256, hidden_dim=128, num_layers=2, num_classes=6)
    model = torch.compile(model, options={"epilogue_fusion": True, "triton.cudagraphs": True})
    print("Model compiled — starting training …")
    
    # Train model
    train_model(model, train_loader, valid_loader, num_epochs=180, device=device)
    # ---------------- Final test on held-out subjects ---------------------
    model.load_state_dict(torch.load("best_sequence_classifier_CT.pth", map_location=device))
    model.eval()
    
    # Standard evaluation (without TTA) for comparison
    all_preds = []
    all_labels = []
    test_correct = test_total = 0
    
    with torch.no_grad():
        for seqs, labs in test_loader:
            seqs, labs = seqs.to(device), labs.to(device)
            with autocast(device_type="cuda"):
                outputs = model(seqs)
            
            preds = outputs.argmax(1)
            test_correct += (preds == labs).sum().item()
            test_total += labs.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labs.cpu().numpy())
    
    # Calculate and print overall test accuracy (without TTA)
    test_acc = test_correct / test_total
    print(f"Standard Test Accuracy (without TTA): {test_acc:.4f}")
    
    # Test-time augmentation evaluation
    print("\nEvaluating with test-time augmentation (5-crop + 2-window)...")
    tta_acc, tta_preds, tta_labels, tta_class_metrics = evaluate_with_tta(
        model, test_dataset, base_dir, device
    )
    
    print(f"Test Accuracy with TTA: {tta_acc:.4f}")
    
    # Calculate per-class metrics
    classes = range(6)  # Assuming 6 classes (0-5)
    print("\nPer-class Accuracy with TTA:")
    
    for cls in classes:
        metrics = tta_class_metrics[cls]
        print(f"Class {cls}: {metrics['accuracy']:.4f} ({metrics['count']} examples)")
    
    # Clean-up temporary CSVs
    os.remove(train_csv)
    os.remove(valid_csv)
    os.remove(test_csv)
if __name__ == "__main__":
    main()