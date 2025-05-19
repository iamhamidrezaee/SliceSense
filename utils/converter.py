import os
import re
import csv
import SimpleITK as sitk
from glob import glob

# ======== CONFIGURATION ========
# Folder containing resampled 3D images
resampled_folder = "/share/sablab/nfs04/users/rs2492/data/nnUNet_preprocessed_DATA/SliceSense/ResampledData_5064"

# Output folder to store JPEG slices folders
jpeg_output_folder = "/share/sablab/nfs04/users/rs2492/data/nnUNet_preprocessed_DATA/SliceSense/JPEGData_5064"
os.makedirs(jpeg_output_folder, exist_ok=True)

# CSV file to store mapping: folder name -> label
csv_mapping_file = os.path.join(jpeg_output_folder, "jpeg_mapping.csv")

# Mapping from target thickness to label
thickness_to_label = {0.5: 0, 1.0: 1, 2.0: 2, 3.0: 3, 5.0: 4, 7.0: 5}

# Regex to extract the slice thickness from the file name:
# e.g. "Totalsegmentator_000001_0.5mm.nii.gz" -> group(1) will be "0.5"
thickness_pattern = re.compile(r"_(\d+(?:\.\d+)?)mm\.nii\.gz$")

# ======== HELPER FUNCTION TO EXTRACT A 2D SLICE FROM A 3D IMAGE ========
def extract_slice(image3d, slice_index):
    """
    Extract a 2D slice from a 3D SimpleITK Image at the given index along the z-axis.
    """
    size = list(image3d.GetSize())
    # Set the size along the z axis to zero for extraction.
    size[2] = 0
    index = [0, 0, slice_index]
    extractor = sitk.ExtractImageFilter()
    extractor.SetSize(size)
    extractor.SetIndex(index)
    slice_image = extractor.Execute(image3d)
    return slice_image

# ======== MAIN PROCESSING ========
# Get list of resampled images
resampled_files = glob(os.path.join(resampled_folder, "*.nii.gz"))
print(f"Found {len(resampled_files)} resampled 3D images.")

# List to hold mapping entries (folder name, label)
mapping_entries = []

for file_path in resampled_files:
    base_filename = os.path.basename(file_path)
    print(f"\nProcessing {base_filename}")
    
    # Extract the thickness value from the filename using regex
    match = thickness_pattern.search(base_filename)
    if not match:
        print(f"[Warning] Could not extract thickness from {base_filename}. Skipping file.")
        continue
    thickness_value = float(match.group(1))
    label = thickness_to_label.get(thickness_value, None)
    if label is None:
        print(f"[Warning] Thickness {thickness_value} not in mapping for {base_filename}. Skipping file.")
        continue

    # Define folder name for JPEG slices: use file name without .nii.gz extension.
    folder_name = os.path.splitext(os.path.splitext(base_filename)[0])[0]
    output_folder = os.path.join(jpeg_output_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Created folder: {output_folder}")

    # Read the 3D image using SimpleITK
    try:
        volume = sitk.ReadImage(file_path)
    except Exception as e:
        print(f"[Error] Could not read {file_path}: {e}")
        continue

    # Verify the image is 3D
    if volume.GetDimension() != 3:
        print(f"[Error] Expected a 3D image but got dimension {volume.GetDimension()} for {file_path}. Skipping.")
        continue

    size = volume.GetSize()  # (width, height, depth)
    num_slices = size[2]
    print(f"Image size: {size}; extracting {num_slices} slices.")

    # Process each slice: extract, rescale intensity, cast to 8-bit, and save as JPEG.
    for i in range(num_slices):
        try:
            slice_img = extract_slice(volume, i)
            # Rescale intensities to [0,255] and cast to unsigned 8-bit.
            slice_img = sitk.RescaleIntensity(slice_img, outputMinimum=0, outputMaximum=255)
            slice_img = sitk.Cast(slice_img, sitk.sitkUInt8)
            
            # Construct output JPEG filename.
            jpeg_filename = f"{folder_name}_frame_{i:03d}.jpg"
            jpeg_path = os.path.join(output_folder, jpeg_filename)
            sitk.WriteImage(slice_img, jpeg_path)
            print(f"  Saved slice {i} as {jpeg_filename}")
        except Exception as e:
            print(f"[Error] Processing slice {i} for {base_filename}: {e}")
            continue

    # Add an entry to the mapping list (folder name and its label).
    mapping_entries.append((folder_name, label))
    print(f"Finished processing {base_filename}. Assigned label: {label}")

# ======== WRITE CSV MAPPING ========
try:
    with open(csv_mapping_file, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["folder_name", "label"])
        for entry in mapping_entries:
            csv_writer.writerow(entry)
    print(f"\nCSV mapping file saved to {csv_mapping_file}")
except Exception as e:
    print(f"[Error] Writing CSV mapping file: {e}")

print("\n=== Conversion to JPEG slices complete ===")
