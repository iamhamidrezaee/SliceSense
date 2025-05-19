import os
import re
import csv
import SimpleITK as sitk
from glob import glob

# ======== CONFIGURATION ========
# Base input directory
base_dir = "DATASET_DIR"
imagesTr_folder = os.path.join(base_dir, "imagesTr")

# Output directory for the resampled 3D images
output_base = "OUTPUT_DIR"
os.makedirs(output_base, exist_ok=True)

# CSV file for mapping image file names to labels (one number per image)
csv_mapping_file = os.path.join(output_base, "training_data.csv")

# Target slice thickness values (in mm) and mapping to unique numeric labels
target_thicknesses = [0.5, 1.0, 2.0, 3.0, 5.0, 7.0]
thickness_to_label = {0.5: 0, 1.0: 1, 2.0: 2, 3.0: 3, 5.0: 4, 7.0: 5}

# Regular expression to extract patient ID and modality (or slice) index from filename (If applicable)
pattern = re.compile(r"YOUR_DATASET_SPECIFIC_EXPRESSION")

# List to collect mapping tuples (output_filename, label)
mapping_list = []

print("=== Starting processing for resampling ===")

# ======== STEP 1: Group Files by Patient ID ========
patient_files = {}
for file_path in glob(os.path.join(imagesTr_folder, "*.nii.gz")):
    file_name = os.path.basename(file_path)
    match = pattern.match(file_name)
    if match:
        patient_id, mod_idx = match.groups()
        patient_files.setdefault(patient_id, []).append((int(mod_idx), file_path))
    else:
        print(f"[Warning] Filename did not match expected pattern: {file_name}")

print(f"Found {len(patient_files)} patients.")

# ======== STEP 2: Process Each Patient's Representative 3D Image ========
for patient_id, files in patient_files.items():
    print(f"\nProcessing patient: {patient_id} with {len(files)} file(s)")
    # Look for file with modality/slice index "0000" (e.g., t1n); if not present, pick the first one.
    selected_file = None
    for mod_idx, fp in files:
        if mod_idx == 0:
            selected_file = fp
            break
    if not selected_file:
        selected_file = sorted(files, key=lambda x: x[0])[0][1]
        print(f"  [Info] Modality 0000 not found; using first file: {selected_file}")
    else:
        print(f"  Selected file for resampling: {selected_file}")
    
    # Read the chosen 3D image using SimpleITK
    try:
        volume_3d = sitk.ReadImage(selected_file)
    except Exception as e:
        print(f"[Error] Failed to read image {selected_file}: {e}")
        continue

    # Print image information for debugging
    print(f"  Image dimension: {volume_3d.GetDimension()}")
    print(f"  Image size: {volume_3d.GetSize()}")
    print(f"  Image spacing: {volume_3d.GetSpacing()}")
    print(f"  Image origin: {volume_3d.GetOrigin()}")
    print(f"  Image direction: {volume_3d.GetDirection()}")

    if volume_3d.GetDimension() != 3:
        print(f"[Error] Expected a 3D image for patient {patient_id} but got dimension {volume_3d.GetDimension()}. Skipping.")
        continue

    # Get the original size and spacing of the 3D image
    orig_size = volume_3d.GetSize()      # (X, Y, Z)
    orig_spacing = volume_3d.GetSpacing()  # (sX, sY, sZ)

    # ======== STEP 3: Resample the 3D Image for Each Target Slice Thickness ========
    # Note: We assume that we want to adjust the spacing in the z axis (index 2) only.
    for target_thickness in target_thicknesses:
        print(f"  Resampling to target slice thickness: {target_thickness}mm")
        new_spacing = list(orig_spacing)
        new_spacing[2] = target_thickness  # update z spacing

        # Compute new z size from the ratio of original z spacing to target thickness.
        new_z = int(round(orig_size[2] * (orig_spacing[2] / target_thickness)))
        new_size = list(orig_size)
        new_size[2] = new_z

        print(f"    Original z spacing: {orig_spacing[2]}, new z spacing: {target_thickness}")
        print(f"    Original z size: {orig_size[2]}, new z size: {new_z}")
        print(f"    New spacing: {new_spacing}, New size: {new_size}")

        # Set up the resampling filter (3D is fully supported)
        resample_filter = sitk.ResampleImageFilter()
        resample_filter.SetOutputSpacing(new_spacing)
        resample_filter.SetSize(new_size)
        resample_filter.SetOutputOrigin(volume_3d.GetOrigin())
        resample_filter.SetOutputDirection(volume_3d.GetDirection())
        # Using linear interpolation for MRI images; for segmentation masks, use nearest neighbor.
        resample_filter.SetInterpolator(sitk.sitkLinear)

        try:
            resampled_volume = resample_filter.Execute(volume_3d)
        except Exception as e:
            print(f"[Error] Resampling failed for patient {patient_id} at thickness {target_thickness}mm: {e}")
            continue

        # ======== STEP 4: Save the Resampled Volume ==========
        thickness_str = f"{target_thickness}mm"
        output_filename = f"{patient_id}_{thickness_str}.nii.gz"
        output_path = os.path.join(output_base, output_filename)
        try:
            sitk.WriteImage(resampled_volume, output_path)
            print(f"    Saved resampled image: {output_path}")
        except Exception as e:
            print(f"[Error] Failed to save image for patient {patient_id} at thickness {target_thickness}mm: {e}")
            continue

        # ======== STEP 5: Record Mapping for CSV ==========
        label = thickness_to_label[target_thickness]
        mapping_list.append((output_filename, label))

# ======== STEP 6: Write the Mapping to CSV ==========
try:
    with open(csv_mapping_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["image_filename", "label"])
        for item in mapping_list:
            csv_writer.writerow(item)
    print(f"\nMapping CSV saved to: {csv_mapping_file}")
except Exception as e:
    print(f"[Error] Failed to write mapping CSV: {e}")

print("\n=== Resampling and mapping complete ===")
