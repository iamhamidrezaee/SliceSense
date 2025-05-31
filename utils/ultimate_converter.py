#!/usr/bin/env python3
"""
Unified BraTS Dataset Processor for SliceSense
==============================================
Processes multiple BraTS datasets on-the-fly:
1. Reads NIfTI files directly from nnUNet raw data
2. Resamples to different slice thicknesses in memory
3. Converts to JPEG slices immediately
4. Only stores final JPEGs and unified CSV mapping

Usage:
    python unified_brats_processor.py
"""

import os
import re
import csv
import SimpleITK as sitk
from glob import glob
from pathlib import Path
import multiprocessing as mp
import numpy as np
from typing import List, Tuple, Dict
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# ======== CONFIGURATION ========
# Base directory containing all datasets
BASE_DATA_DIR = "/share/sablab/nfs04/users/rs2492/data/nnUNet_preprocessed_DATA/nnUNet_raw_data_base"

# List of datasets to process
DATASETS = [
    "Dataset5000_BraTS-GLI_2023",
    "Dataset5001_BraTS-SSA_2023", 
    "Dataset5002_BraTS-MEN_2023",
    "Dataset5003_BraTS-MET_2023",
    "Dataset5007_UCSF-BMSR",
    "Dataset5010_ATLASR2",
    "Dataset5083_IXIT1",
    "Dataset5084_IXIT2", 
    "Dataset5085_IXIPD",
    "Dataset5141_NYU-METS",
    "Dataset5113_StanfordMETShare",
    "Dataset5018_TopCoWMRAwholeBIN"
]

# Output directory for unified JPEG dataset
OUTPUT_DIR = "SliceSense_Unified_Dataset"
JPEG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "jpeg_slices")
CSV_MAPPING_FILE = os.path.join(OUTPUT_DIR, "unified_mapping.csv")

# Target slice thicknesses and labels
TARGET_THICKNESSES = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
THICKNESS_TO_LABEL = {0.5: 0, 1.0: 1, 2.0: 2, 3.0: 3, 5.0: 4, 7.0: 5}

# Preferred modality priority for BraTS datasets
MODALITY_PRIORITY = ["t1ce", "t1", "t2", "flair"]

# Number of CPU cores to use
N_CORES = min(20, mp.cpu_count())

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('unified_processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ======== HELPER FUNCTIONS ========

def find_best_modality_file(patient_files: List[str]) -> str:
    """
    Find the best modality file based on priority.
    BraTS files typically end with _0000.nii.gz (t1), _0001.nii.gz (t1ce), etc.
    """
    modality_map = {}
    
    for file_path in patient_files:
        filename = os.path.basename(file_path)
        
        # Extract modality info from filename
        if "_0000.nii.gz" in filename:  # t1
            modality_map["t1"] = file_path
        elif "_0001.nii.gz" in filename:  # t1ce
            modality_map["t1ce"] = file_path
        elif "_0002.nii.gz" in filename:  # t2
            modality_map["t2"] = file_path
        elif "_0003.nii.gz" in filename:  # flair
            modality_map["flair"] = file_path
    
    # Select based on priority
    for modality in MODALITY_PRIORITY:
        if modality in modality_map:
            return modality_map[modality]
    
    # Fallback to first file if no standard modality found
    return patient_files[0]

def extract_patient_id(filepath: str, dataset_name: str) -> str:
    """Extract patient ID from filepath, handling different naming conventions."""
    filename = os.path.basename(filepath)
    
    # Remove dataset-specific prefixes and suffixes
    # Most BraTS files follow pattern: [DatasetName]_[PatientID]_[modality].nii.gz
    base_name = filename.replace(".nii.gz", "").replace("_0000", "").replace("_0001", "").replace("_0002", "").replace("_0003", "")
    
    # Create unique patient ID with dataset prefix to avoid conflicts
    patient_id = f"{dataset_name}_{base_name}"
    
    return patient_id

def resample_volume(volume: sitk.Image, target_thickness: float) -> sitk.Image:
    """Resample 3D volume to target slice thickness."""
    orig_spacing = volume.GetSpacing()
    orig_size = volume.GetSize()
    
    # Create new spacing with updated z-axis
    new_spacing = list(orig_spacing)
    new_spacing[2] = target_thickness
    
    # Calculate new size
    new_size = list(orig_size)
    new_size[2] = int(round(orig_size[2] * (orig_spacing[2] / target_thickness)))
    
    # Setup resampler
    resample_filter = sitk.ResampleImageFilter()
    resample_filter.SetOutputSpacing(new_spacing)
    resample_filter.SetSize(new_size)
    resample_filter.SetOutputOrigin(volume.GetOrigin())
    resample_filter.SetOutputDirection(volume.GetDirection())
    resample_filter.SetInterpolator(sitk.sitkLinear)
    
    return resample_filter.Execute(volume)

def extract_slice_2d(image3d: sitk.Image, slice_index: int) -> sitk.Image:
    """Extract 2D slice from 3D image."""
    size = list(image3d.GetSize())
    size[2] = 0  # Set z-size to 0 for 2D extraction
    
    index = [0, 0, slice_index]
    
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size)
    extractor.SetIndex(index)
    
    return extractor.Execute(image3d)

def process_single_file(args: Tuple[str, str, str]) -> List[Tuple[str, int]]:
    """
    Process a single NIfTI file: resample and convert to JPEGs.
    Returns list of (folder_name, label) tuples for CSV mapping.
    """
    file_path, dataset_name, patient_id = args
    
    try:
        logger.info(f"Processing {patient_id} from {dataset_name}")
        
        # Read 3D volume
        volume = sitk.ReadImage(file_path)
        
        if volume.GetDimension() != 3:
            logger.warning(f"Skipping non-3D image: {file_path}")
            return []
        
        mapping_entries = []
        
        # Process each target thickness
        for target_thickness in TARGET_THICKNESSES:
            try:
                # Resample volume in memory
                resampled_volume = resample_volume(volume, target_thickness)
                
                # Create output folder for this thickness
                thickness_str = f"{target_thickness}mm"
                folder_name = f"{patient_id}_{thickness_str}"
                output_folder = os.path.join(JPEG_OUTPUT_DIR, folder_name)
                os.makedirs(output_folder, exist_ok=True)
                
                # Extract and save all slices as JPEGs
                num_slices = resampled_volume.GetSize()[2]
                
                for slice_idx in range(num_slices):
                    # Extract 2D slice
                    slice_2d = extract_slice_2d(resampled_volume, slice_idx)
                    
                    # Rescale intensity and convert to 8-bit
                    slice_2d = sitk.RescaleIntensity(slice_2d, outputMinimum=0, outputMaximum=255)
                    slice_2d = sitk.Cast(slice_2d, sitk.sitkUInt8)
                    
                    # Save as JPEG
                    jpeg_filename = f"{folder_name}_slice_{slice_idx:03d}.jpg"
                    jpeg_path = os.path.join(output_folder, jpeg_filename)
                    sitk.WriteImage(slice_2d, jpeg_path)
                
                # Add to mapping
                label = THICKNESS_TO_LABEL[target_thickness]
                mapping_entries.append((folder_name, label))
                
                logger.info(f"Processed {patient_id} at {thickness_str}: {num_slices} slices")
                
            except Exception as e:
                logger.error(f"Error processing {patient_id} at {target_thickness}mm: {e}")
                continue
        
        return mapping_entries
        
    except Exception as e:
        logger.error(f"Error processing file {file_path}: {e}")
        return []

def collect_all_files() -> List[Tuple[str, str, str]]:
    """
    Collect all NIfTI files from all datasets.
    Returns list of (file_path, dataset_name, patient_id) tuples.
    """
    all_files = []
    
    for dataset_name in DATASETS:
        dataset_path = os.path.join(BASE_DATA_DIR, dataset_name)
        images_tr_path = os.path.join(dataset_path, "imagesTr")
        
        if not os.path.exists(images_tr_path):
            logger.warning(f"Dataset path not found: {images_tr_path}")
            continue
        
        # Find all NIfTI files
        nifti_files = glob(os.path.join(images_tr_path, "*.nii.gz"))
        logger.info(f"Found {len(nifti_files)} files in {dataset_name}")
        
        # Group files by patient (handle multiple modalities)
        patient_groups = {}
        for file_path in nifti_files:
            # Extract base patient ID (remove modality suffix)
            base_name = os.path.basename(file_path)
            # Remove modality indicators (_0000, _0001, etc.)
            patient_base = re.sub(r'_\d{4}\.nii\.gz$', '', base_name)
            
            if patient_base not in patient_groups:
                patient_groups[patient_base] = []
            patient_groups[patient_base].append(file_path)
        
        # Select best modality file for each patient
        for patient_base, patient_files in patient_groups.items():
            best_file = find_best_modality_file(patient_files)
            patient_id = extract_patient_id(best_file, dataset_name)
            all_files.append((best_file, dataset_name, patient_id))
    
    logger.info(f"Total files to process: {len(all_files)}")
    return all_files

def main():
    """Main processing pipeline."""
    start_time = time.time()
    
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(JPEG_OUTPUT_DIR, exist_ok=True)
    
    logger.info("Starting unified BraTS dataset processing...")
    logger.info(f"Processing {len(DATASETS)} datasets with {N_CORES} CPU cores")
    
    # Collect all files to process
    all_files = collect_all_files()
    
    if not all_files:
        logger.error("No files found to process!")
        return
    
    # Process files in parallel
    all_mapping_entries = []
    
    logger.info(f"Starting parallel processing of {len(all_files)} files...")
    
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(process_single_file, file_args): file_args 
            for file_args in all_files
        }
        
        # Collect results as they complete
        completed = 0
        for future in as_completed(future_to_file):
            file_args = future_to_file[future]
            try:
                mapping_entries = future.result()
                all_mapping_entries.extend(mapping_entries)
                completed += 1
                
                if completed % 10 == 0:
                    logger.info(f"Completed {completed}/{len(all_files)} files")
                    
            except Exception as e:
                logger.error(f"Error processing {file_args[2]}: {e}")
    
    # Write unified CSV mapping
    logger.info(f"Writing unified CSV mapping with {len(all_mapping_entries)} entries...")
    
    try:
        with open(CSV_MAPPING_FILE, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['folder_name', 'label'])
            for entry in all_mapping_entries:
                writer.writerow(entry)
        
        logger.info(f"CSV mapping saved to: {CSV_MAPPING_FILE}")
        
    except Exception as e:
        logger.error(f"Error writing CSV file: {e}")
    
    # Summary statistics
    end_time = time.time()
    processing_time = end_time - start_time
    
    logger.info("=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Total processing time: {processing_time:.2f} seconds ({processing_time/60:.2f} minutes)")
    logger.info(f"Files processed: {len(all_files)}")
    logger.info(f"JPEG folders created: {len(all_mapping_entries)}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info(f"JPEG slices directory: {JPEG_OUTPUT_DIR}")
    logger.info(f"CSV mapping file: {CSV_MAPPING_FILE}")
    
    # Count by label
    label_counts = {}
    for _, label in all_mapping_entries:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info("\nLabel distribution:")
    for thickness, label in THICKNESS_TO_LABEL.items():
        count = label_counts.get(label, 0)
        logger.info(f"  {thickness}mm (label {label}): {count} folders")

if __name__ == "__main__":
    main()