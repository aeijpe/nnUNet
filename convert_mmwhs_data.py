import os
import glob
import numpy as np
import nibabel as nib
import re

def rename(directory_a, directory_b, r1, r2):
    directory_b1 = os.path.join(directory_b, "imagesTr")
    directory_b2 = os.path.join(directory_b, "labelsTr")

    for i in range(r1, r2):
        images = sorted(glob.glob(os.path.join(directory_a, f"img{i}_slice*.nii.gz")))
        labels = sorted(glob.glob(os.path.join(directory_a, f"lab{i}_slice*.nii.gz")))

        # Iterate over all files
        it = 0
        for image, label in zip(images, labels):
            # Check if it is a file (not a directory)
            new_filename = f"WHSMR_{j}x{it}_0000.nii.gz"
            if os.path.isfile(image):
                # Construct the new file name
                new_path = os.path.join(directory_b1, new_filename)
                # Rename the file
                os.rename(image, new_path)

            if os.path.isfile(label):
                # Construct the new file name
                new_path = os.path.join(directory_b2, new_filename)
                # Rename the file
                os.rename(label, new_path)
            
            it += 1 
        j += 1
        print("Files have been renamed successfully.")

def map_labels(directory_b):
    directory_b2 = os.path.join(directory_b, "labelsTrAll")

    labels = sorted(glob.glob(os.path.join(directory_b2, "*.nii.gz")))
    # Define the mapping of old values to new values
    label_mapping = {
        0.: 0.,
        1.: 1.,
        2.: 0.,
        3.: 2.,
        4.: 0.,
        5.: 3.
    }

    i = 0

    os.makedirs(os.path.join(directory_b, "labelsTr"), exist_ok=True)
    for file in labels:
        # Use re.sub to replace 'labelsTrOld' with 'labelsTr'
        new_path = re.sub(r'labelsTrAll', 'labelsTr', file)
        # Load the NIfTI file
        lab = nib.load(file)
        data = lab.get_fdata()
        # Replace the values in the data array
        for old_value, new_value in label_mapping.items():
            data[np.isclose(data, old_value)] = new_value
        # Create a new NIfTI image with the modified data
        data = data.squeeze(0)[:, :, np.newaxis]
        new_lab = nib.Nifti1Image(data, lab.affine, lab.header)

        # Save the new NIfTI image to a file
        nib.save(new_lab, new_path)

def center_crop(data, target_height, target_width):
    """Crop the center of the 3D image data to the target height and width."""
    _, height, width = data.shape
    start_y = (height - target_height) // 2
    start_x = (width - target_width) // 2
    return data[:, start_y:start_y + target_height, start_x:start_x + target_width]

def crop(directory_b):
    images_file_path = os.path.join(directory_b, "imagesTr")
    labels_file_path = os.path.join(directory_b, "labelsTr3")

    images = sorted(glob.glob(os.path.join(images_file_path, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(labels_file_path, "*.nii.gz")))

    os.makedirs(os.path.join(directory_b, "imagesTrPr"), exist_ok=True)
    os.makedirs(os.path.join(directory_b, "labelsTrPr"), exist_ok=True)

    # Cropping is not an affine transform! Header image dimensions get updated automatically.
    for image_file, label_file in zip(images, labels):
        print('hier')
        image = nib.load(image_file)
        label = nib.load(label_file)

        image_data = image.get_fdata()
        label_data = label.get_fdata()

        # Perform center crop
        cropped_data = center_crop(image_data, 220, 220)
        # Create a new NIfTI image with the cropped data
        cropped_img = nib.Nifti1Image(cropped_data, image.affine, image.header)

        new_path_image = re.sub(r'imagesTr', 'imagesTrPr', image_file)
        nib.save(cropped_img, new_path_image)

        # Perform center crop
        cropped_label_data = center_crop(label_data, 220, 220)
        # Create a new NIfTI image with the cropped data
        cropped_label = nib.Nifti1Image(cropped_label_data, label.affine, label.header)

        new_path_label = re.sub(r'labelsTr3', 'labelsTrPr', label_file)
        nib.save(cropped_label, new_path_label)


def change_dim(dir):
    images_file_path = os.path.join(dir, "imagesTr")
    labels_file_path = os.path.join(dir, "labelsTr")

    images = sorted(glob.glob(os.path.join(images_file_path, "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(labels_file_path, "*.nii.gz")))

    i = 0
    for image_file, label_file in zip(images, labels):
        
        # image = nib.load(image_file)
        label = nib.load(label_file)

        # image_data = image.get_fdata()
        label_data = label.get_fdata()

        # print(image_data.shape)
        # print(label_data.shape)

        # image_data = image_data.squeeze(0)[:, :, np.newaxis]
        label_data = label_data.squeeze(0)[:, :, np.newaxis]

        # new_img = nib.Nifti1Image(image_data, image.affine, image.header)
        # nib.save(new_img, image_file)

        new_lab = nib.Nifti1Image(label_data, label.affine, label.header)
        nib.save(new_lab, label_file)
        i += 1

    print(f"Done: i = {i}")


def rename_folds(dir1, dir2, cases, fold):
        output_dir = os.path.join(dir2, f"fold_{fold}/")
        os.makedirs(output_dir, exist_ok=True)
        img_files = []
        for i in cases:
            img_files += glob.glob(os.path.join(dir1, f"WHSMR_{i}x*.nii.gz"))

        img_files = sorted(img_files)

        # Iterate over all files
        it = 0
        for image in img_files:
            # Check if it is a file (not a directory)
            new_path = re.sub(dir1, output_dir, image)
            # Rename the file
            os.rename(image, new_path)
        
        print("Files have been renamed successfully.")



def main():
    # Specify the directory containing the files
    # directory_a = "/data/groups/public/archive/radiology/multimodal_raw/Dataset033_MMWHSCT/imagesTrCopy"
    # directory_b = "/data/groups/public/archive/radiology/multimodal_raw/Dataset033_MMWHSCT/imagesNA"
    # # CT: r1 == 33, r2 == 53
    # # MRI: r1 == 1, r2 == 21
    np.random.seed(42)

    # print("mapping labels")
    # map_labels(directory_b)

    # print("cropping")
    # crop(directory_b)

    # print("Change dim")
    # change_dim(directory_b)
    print("remove_dir")
    # rename_folds(directory_a, directory_b, [12, 13, 15, 19], 0)
    # rename_folds(directory_a, directory_b, [7, 8, 11, 16], 1)
    # rename_folds(directory_a, directory_b, [3, 14, 17, 18], 2)
    # rename_folds(directory_a, directory_b, [4, 5, 6, 10], 3)
    # rename_folds(directory_a, directory_b, [0, 1, 2, 9], 4)


    directory_a = "/data/groups/public/archive/radiology/multimodal_raw/Dataset034_MMWHSMRI/imagesTrCopy"
    directory_b = "/data/groups/public/archive/radiology/multimodal_raw/Dataset034_MMWHSMRI/imagesNA"
    rename_folds(directory_a, directory_b, [6, 9, 13, 16], 0)
    rename_folds(directory_a, directory_b, [1, 2, 4, 18], 1)
    rename_folds(directory_a, directory_b, [7, 12, 14, 17], 2)
    rename_folds(directory_a, directory_b, [5, 10, 11, 19], 3)
    rename_folds(directory_a, directory_b, [0, 3, 8, 15], 4)



if __name__ == "__main__":
    main()