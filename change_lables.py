import os
import nibabel as nib
import numpy as np

# Define the folder containing the NIfTI files
input_folder = "./masks"


# Define the label mapping
label_map = {
    7: 3,
    5: 1,
    6: 2,
    8: 4
}

for filename in os.listdir(input_folder):
    if filename.endswith(".nii") or filename.endswith(".nii.gz"):
        file_path = os.path.join(input_folder, filename)
        
        # Load the image
        img = nib.load(file_path)
        data = img.get_fdata().astype(np.int32)  # <<< Force integer type

        # Initialize a copy of the data
        converted_data = np.copy(data)

        # Apply mapping without collision issues
        for original, new in label_map.items():
            converted_data[data == original] = new

        # Save with a new name
        new_filename = filename.replace(".nii", "_converted.nii").replace(".nii.gz", "_converted.nii.gz")
        new_path = os.path.join(input_folder, new_filename)

        new_img = nib.Nifti1Image(converted_data, affine=img.affine, header=img.header)
        nib.save(new_img, new_path)

        print(f"Converted and saved: {new_filename}")