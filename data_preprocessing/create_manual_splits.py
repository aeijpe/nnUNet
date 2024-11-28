import json
import os
from sklearn.model_selection import KFold
import random
import glob
import numpy as np
import torch
import argparse

def get_cases(train, val, test, all_cases_dir, data_type):
    train_cases = []
    val_cases = []
    test_cases = []
    
    # template = '{data_type}_{person}x{slide}'
    for item in train:
        for i in range(len(glob.glob(os.path.join(all_cases_dir, f"{data_type}_{item}x*.nii.gz")))):
            train_cases.append(f'{data_type}_{item}x{i}')

    for item in val:
        # per case we have 16 2d images
        for i in range(len(glob.glob(os.path.join(all_cases_dir, f"{data_type}_{item}x*.nii.gz")))):
            val_cases.append(f'{data_type}_{item}x{i}')

    for item in test:
        # per case we have 16 2d images
        for i in range(len(glob.glob(os.path.join(all_cases_dir, f"{data_type}_{item}x*.nii.gz")))):
            test_cases.append(f'{data_type}_{item}x{i}')

    return train_cases, val_cases, test_cases


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cases = range(0,20)
    kf = KFold(n_splits=5, shuffle=True, random_state=1234) # Same as nnUNet
    folds = []
    already_used = []

    all_cases_dir = os.path.join(args.data_dir, "gt_segmentations")
    for i, (fold_trainval, fold_test) in enumerate(kf.split(cases)):
        print(f"Fold {i}")
        random.shuffle(fold_trainval)
        cases_not_used = [elem for elem in fold_trainval if elem not in already_used]
        assert len(cases_not_used) >= 3 
  
        fold_val = cases_not_used[:3]
        fold_train = [elem for elem in fold_trainval if elem not in fold_val]
        already_used += fold_val
        
        train_cases, val_cases, test_cases = get_cases(fold_train, fold_val, fold_test, all_cases_dir, args.data_type)
        folds.append({ # Fold 1
            "train": train_cases,
            "val": val_cases, 
            "test": test_cases
        })

    # Define the full path to the output file
    output_file_path = os.path.join(args.data_dir, "splits_final.json")

    # Save the dictionary to a JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(folds, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create manual splits')
    parser.add_argument('--data_dir', type=str, default='nnunetv2/nnUNet_preprocessed/Dataset035_CHAOST1', help='Data directory')
    parser.add_argument('--data_type', type=str, default=None, help='raw data directory: Choose between CHAOST1, CHAOST2, MMWHSMRI, MMWHSCT')
    parser.add_argument('--seed', type=int, default=1, help='seed')
    args = parser.parse_args()

    main(args)