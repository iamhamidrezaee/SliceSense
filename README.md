<p align="center">
  <img src="utils/SliceSense_logo.png" alt="SliceSense Logo" width="500"/>
</p>

<p align="center">
  <b>Deep learning-powered slice thickness prediction for medical imaging</b>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#key-features">Key Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#installation">Installation</a> •
  <a href="#usage">Usage</a> •
  <a href="#training">Training</a> •
  <a href="#utilities">Utilities</a> •
  <a href="#license">License</a>
</p>

## Overview

SliceSense is a deep learning framework for predicting the z-axis slice thickness of 3D medical images (MRI/CT) from their 2D slices. This information is crucial for numerous medical imaging applications, clinical diagnostics, and research scenarios where slice thickness affects analysis accuracy.

Using a specially designed neural network architecture combining ResNet feature extraction, bidirectional LSTM, and self-attention mechanisms, SliceSense achieves high accuracy in classifying slice thickness across common radiological imaging standards (0.5mm, 1mm, 2mm, 3mm, 5mm, and 7mm).

## Key Features

- **Advanced Architecture**: CNN + BiLSTM + Self-Attention for optimal sequence modeling of medical slices
- **Modality Support**: Specialized models for both CT and MRI imaging
- **Sliding Window Inference**: Robust prediction through overlapping slice sequences
- **3D Reconstruction**: Automatic reconstruction of NIfTI images with predicted slice thickness
- **Test-Time Augmentation**: Enhanced accuracy through multi-crop and window ensemble techniques
- **Optimized Performance**: CUDA acceleration, mixed precision training, and torch.compile support

## 🧠  Architecture

SliceSense employs a multi-stage deep learning architecture:

1. **Feature Extraction**: ResNet-18 backbone (pre-trained on ImageNet) processes each 2D slice
2. **Feature Projection**: Linear projection with normalization reduces dimensionality
3. **Sequence Modeling**: Bidirectional LSTM captures relationships between adjacent slices
4. **Self-Attention**: Multi-head attention mechanism highlights important slice relationships
5. **Classification**: MLP head with regularization predicts the slice thickness class

## How It Works

SliceSense processes medical imaging data through several key steps:

1. **Preprocessing**: 3D volumes (DICOM/NIfTI) are split into 2D slices and saved as JPEG
2. **Inference**: 
   - Slices are processed through a sliding window approach (10 slices per window)
   - Each window receives a classification prediction
   - Multiple windows are aggregated using majority voting
3. **Reconstruction**: Original 3D volume is reconstructed with predicted slice thickness metadata
4. **Output**: Annotated NIfTI file with correct z-axis spacing information

## Installation

```bash
# Clone the repository
git clone https://github.com/iamhamidrezaee/slicesense.git
cd slicesense

# Create conda environment
conda create -n slicesense python=3.10
conda activate slicesense

# Install dependencies
pip install torch torchvision torchaudio
pip install nibabel pandas SimpleITK scikit-learn matplotlib
```

## Usage

### Inference on a Directory of 2D Slices

```python
from SliceSenseInference import slicesense_inference

# Path to model file
model_path = "trained_models/best_sequence_classifier_MRI.pth"

# Path to directory containing JPEG slices
input_dir = "path/to/slices_folder"

# Run inference
output_path = slicesense_inference(model_path, input_dir)
print(f"Reconstructed 3D volume saved to: {output_path}")
```

### Programmatic API

```python
import torch
from SliceSenseInference import load_model, sliding_window_inference

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model("trained_models/best_sequence_classifier_CT.pth", device)

# Get image paths
import glob
image_paths = sorted(glob.glob("path/to/slices/*.jpg"))

# Run inference
predicted_class, probabilities = sliding_window_inference(
    model, image_paths, window_size=10, stride=5, device=device
)

# Map class to thickness
thickness_map = {0: 0.5, 1: 1.0, 2: 2.0, 3: 3.0, 4: 5.0, 5: 7.0}
print(f"Predicted thickness: {thickness_map[predicted_class]}mm")
print(f"Confidence: {probabilities[predicted_class]:.2%}")
```

## 🏗️ Training

SliceSense provides specialized training scripts for both CT and MRI data:

### CT Training

```bash
# Navigate to Trainers directory
cd Trainers

# Run CT training script
python Trainer_CT.py
```

### MRI Training

```bash
# Run MRI training script
python Trainer_MRI.py
```

### Training Features

- **Dataset Management**: Subject-based train/validation/test splits
- **Augmentation**: Modality-specific data augmentation pipelines
- **Optimization**: Mixed precision, AdamW optimizer, cosine learning rate schedule
- **Monitoring**: Real-time loss/accuracy curves with matplotlib
- **Evaluation**: Test-time augmentation with 5-crop and multi-window techniques

## Utilities

SliceSense includes utilities for dataset preparation:

### 3D Image Resampling

The `utils/sampling.py` script resamples 3D volumes to various slice thicknesses:

```bash
# Configure input/output paths in the script, then run:
python utils/sampling.py
```

### JPEG Conversion

The `utils/converter.py` script converts 3D NIfTI volumes to 2D JPEG slices:

```bash
# Configure paths in the script, then run:
python utils/converter.py
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Totalsegmentator](https://github.com/wasserth/TotalSegmentator) for the dataset
- [PyTorch](https://pytorch.org/) for the deep learning framework
- [SimpleITK](https://simpleitk.org/) for medical image processing