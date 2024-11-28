import os
import glob
import numpy as np
import nibabel as nib
import re
import pydicom
from scipy.ndimage import zoom
from PIL import Image
# os.environ['ITK_NIFTI_SFORM_PERMISSIVE'] = '1'
import SimpleITK as sitk
import shutil
import dicom2nifti
import SimpleITK as sitk
import argparse

def convert_MR_seg(loaded_png):
    result = np.zeros(loaded_png.shape)
    result[(loaded_png > 55) & (loaded_png <= 70)] = 1 # liver
    result[(loaded_png > 110) & (loaded_png <= 135)] = 2 # right kidney
    result[(loaded_png > 175) & (loaded_png <= 200)] = 3 # left kidney
    result[(loaded_png > 240) & (loaded_png <= 255)] = 4 # spleen
    return result

def copy_geometry(image: sitk.Image, ref: sitk.Image):
    image.SetOrigin(ref.GetOrigin())
    image.SetDirection(ref.GetDirection())
    image.SetSpacing(ref.GetSpacing())
    return image

def load_png_stack(folder):
    pngs = os.listdir(folder)
    pngs.sort()
    loaded = []
    for p in pngs:
        loaded.append(np.array(Image.open(os.path.join(folder, p))))
    loaded = np.stack(loaded, 0)[::-1]
    return loaded

def preprocess(args, data_raw_name, img_raw_folder):
    directory_img = os.path.join(args.folder_pp_data, "imagesTr")
    directory_lab = os.path.join(args.folder_pp_data, "labelsTr")
    os.makedirs(directory_img, exist_ok=True)
    os.makedirs(directory_lab, exist_ok=True)
    patient_ids = []
    
    # Process 
    d = os.path.join(args.folder_raw_data, "MR")

    for i, person in enumerate(os.listdir(d)):
        patient_name = f"{args.data_type}_{i}"
        person_path = os.path.join(d, person)
        
        # LABEL
        gt_dir = os.path.join(os.path.join(person_path, data_raw_name), "Ground")
        seg = convert_MR_seg(load_png_stack(gt_dir)[::-1])
        label_out_file = os.path.join(directory_lab, patient_name + ".nii.gz")
        
        # IMAGE
        img_dir = os.path.join(person_path, img_raw_folder)
        img_outfile = os.path.join(directory_img, patient_name + "_0000.nii.gz")
        _ = dicom2nifti.convert_dicom.dicom_series_to_nifti(img_dir, img_outfile, reorient_nifti=False)

        img_sitk = sitk.ReadImage(img_outfile)
        seg_itk = sitk.GetImageFromArray(seg.astype(np.uint8))
        seg_itk = copy_geometry(seg_itk, img_sitk)
        sitk.WriteImage(seg_itk, label_out_file)
        patient_ids.append(patient_name)


def main():
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
    parser = argparse.ArgumentParser(description='CHAOS preproces Script V2')
    parser.add_argument('--folder_raw_data', type=str, default='/data/groups/beets-tan/a.eijpe/multimodal_raw/chaos', help='raw data directory')
    parser.add_argument('--folder_pp_data', type=str, default='/data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset035_CHAOST1', help='preprocessed data directory')
    parser.add_argument('--data_type', type=str, default=None, help='raw data directory: Choose between CHAOST1 and CHAOST2')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    args = parser.parse_args()

    main(args)