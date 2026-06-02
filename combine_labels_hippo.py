import os
import nibabel as nib
import numpy as np

# Path to the parent folder containing subfolders with segmentations
base_dir = "./segs"

# Loop through each subject folder
for folder in os.listdir(base_dir):
    sub_dir = os.path.join(base_dir, folder)
    if not os.path.isdir(sub_dir):
        continue

    # Search for LH and RH segmentation files
    lh_file = None
    rh_file = None
    for f in os.listdir(sub_dir):
        if f.endswith("_LH_seg.nii.gz"):
            lh_file = os.path.join(sub_dir, f)
        elif f.endswith("_RH_seg.nii.gz"):
            rh_file = os.path.join(sub_dir, f)

    if lh_file is None or rh_file is None:
        print(f"Skipping {folder}: Missing LH or RH segmentation.")
        continue

    # Load segmentations
    lh_img = nib.load(lh_file)
    rh_img = nib.load(rh_file)
    lh_data = (nib.load(lh_file).get_fdata() > 0).astype(np.uint8)  # Label as 1
    rh_data = (nib.load(rh_file).get_fdata() > 0).astype(np.uint8)  # Label as 2

    # Combine into a single mask
    combined = np.zeros_like(lh_data, dtype=np.uint8)
    combined[lh_data == 1] = 1
    combined[rh_data == 1] = 2  # This will overwrite if there's overlap

    # Save the combined segmentation
    combined_img = nib.Nifti1Image(combined, affine=lh_img.affine, header=lh_img.header)
    combined_filename = os.path.join(sub_dir, folder + "_combined_hippo_seg.nii.gz")
    nib.save(combined_img, combined_filename)

    print(f"Saved combined segmentation: {combined_filename}")
