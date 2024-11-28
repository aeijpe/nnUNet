import os
import glob
import numpy as np
import nibabel as nib
import re
import pydicom
from scipy.ndimage import zoom
from PIL import Image
os.environ['ITK_NIFTI_SFORM_PERMISSIVE'] = '1'
import SimpleITK as sitk
import shutil
import argparse

def center_crop(data, target_height, target_width):
    """Crop the center of the 3D image data to the target height and width."""
    height, width, _ = data.shape
    start_y = (height - target_height) // 2
    start_x = (width - target_width) // 2
    return data[start_y:start_y + target_height, start_x:start_x + target_width, :]

def preprocess(args, data_raw_name, img_raw_folder):
    directory_img = os.path.join(args.folder_pp_data, "imagesTr")
    directory_lab = os.path.join(args.folder_pp_data, "labelsTr")
    os.makedirs(directory_img, exist_ok=True)
    os.makedirs(directory_lab, exist_ok=True)
    label_mapping = {
        0: 0,
        63: 1,
        126: 2,
        189: 3,
        252: 4
    }

    for i, person in enumerate(sorted(glob.glob(os.path.join(os.path.join(args.folder_raw_data, "MR"), "*")))):
        person_img_files = sorted(glob.glob(os.path.join(os.path.join(person, img_raw_folder), "*.dcm")))
        person_lab_files = sorted(glob.glob(os.path.join(os.path.join(person, data_raw_name), "Ground/*.png")))
        for it, (img, lab) in enumerate(zip(person_img_files, person_lab_files)):
            ds = pydicom.dcmread(img)
            data = ds.pixel_array
            label = Image.open(lab)
            label_data = np.array(label)

            # Get DICOM metadata for affine matrix
            voxel_size = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]), float(ds.SliceThickness))
            orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
            position = np.array(ds.ImagePositionPatient)

            # Calculate the affine matrix
            affine = np.eye(4)
            affine[:3, :2] = orientation.T * voxel_size[:2]
            affine[:3, 3] = position

            # Ensure data is 3D
            if len(data.shape) == 2:
                data = data[:, :, np.newaxis]

            # Rescale voxel sizes to 1x1 for x and y axes, maintaining the original slice thickness
            new_voxel_size = (2.0, 2.0, voxel_size[2])
            zoom_factors = ((new_voxel_size[0] / voxel_size[0]), (new_voxel_size[1] / voxel_size[1]), 1)
            resampled_data = zoom(data, zoom_factors, order=1)  # Use linear interpolation For nearest interpolation, use order = 0 (labels!)

            # Adjust the affine matrix to reflect the new voxel sizes
            new_affine = np.copy(affine)
            new_affine[0, 0] = new_voxel_size[0]
            new_affine[1, 1] = new_voxel_size[1]
            new_affine[2, 2] = voxel_size[2]  # Slice thickness remains the same
            new_affine[0, 3] = 0.0
            new_affine[1, 3] = 0.0
            new_affine[2, 3] = 0.0


            # resampled_data = center_crop(resampled_data, 220, 220)

            # Create a NIfTI image with resampled data and new affine
            nifti_img = nib.Nifti1Image(resampled_data, new_affine)

            # Copy relevant metadata from DICOM to NIfTI header
            header = nifti_img.header
            header['pixdim'][1:4] = new_voxel_size

            # Ensure label data is 3
            if len(label_data.shape) == 2:
                label_data = label_data[ :, :, np.newaxis]  # Add singleton dimension for Z
            
            # Resample the label data to achieve the new voxel sizes using nearest neighbor interpolation
            resampled_label_data = zoom(label_data, zoom_factors, order=0)  # Nearest neighbor interpolation for labels

            for old_value, new_value in label_mapping.items():
                resampled_label_data[resampled_label_data == old_value] = new_value

            # resampled_label_data = center_crop(resampled_label_data, 220, 220)

            # Create a NIfTI image with resampled label data and the same affine
            label_nifti_img = nib.Nifti1Image(resampled_label_data, new_affine)
            # Update NIfTI header to reflect new voxel sizes
            label_header = label_nifti_img.header
            label_header['pixdim'][1:4] = new_voxel_size


            # Save the NIfTI image as a .nii.gz file
            output_filename = os.path.join(directory_img, f"{args.data_type}_{i}x{it}_0000.nii.gz")
            label_output_filename = os.path.join(directory_lab, f"{args.data_type}_{i}x{it}.nii.gz")
            nib.save(nifti_img, output_filename)
            nib.save(label_nifti_img, label_output_filename)

def main(args):
    # Specify the directory containing the files
    np.random.seed(args.seed)
    if args.data_type == "CHAOST1":
        data_raw_name = "T1DUAL"
        img_raw_folder = os.path.join(data_raw_name, "DICOM_anon/inPhase/")
    elif args.data_type == "CHAOST1":
        data_raw_name = "T2SPIR"
        img_raw_folder = os.path.join(data_raw_name, "DICOM_anon/")
    else:
        print("Data type is not supported")

    preprocess(args, data_raw_name, img_raw_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CHAOS preprocess Script')
    parser.add_argument('--folder_raw_data', type=str, default='/data/groups/beets-tan/a.eijpe/multimodal_raw/chaos', help='raw data directory')
    parser.add_argument('--folder_pp_data', type=str, default='/data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset035_CHAOST1', help='preprocessed data directory')
    parser.add_argument('--data_type', type=str, default=None, help='raw data directory: Choose between CHAOST1 and CHAOST2')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()

    main(args)