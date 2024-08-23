import json
import os
from sklearn.model_selection import KFold
import random



def get_cases(train, val, test):
    train_cases = []
    val_cases = []
    test_cases = []
    # template = 'WHSMR_0x1'
    for item in train:
        # per case we have 16 2d images
        for i in range(16):
            train_cases.append(f'WHSCT_{item}x{i}')

    for item in val:
        # per case we have 16 2d images
        for i in range(16):
            val_cases.append(f'WHSCT_{item}x{i}')

    for item in test:
        # per case we have 16 2d images
        for i in range(16):
            test_cases.append(f'WHSCT_{item}x{i}')

    return train_cases, val_cases, test_cases


def main():
    cases = range(0,20)
    kf = KFold(n_splits=5, shuffle=True)
    folds = []
    already_used = []
    for i, (fold_trainval, fold_test) in enumerate(kf.split(cases)):
        print(f"Fold {i}")
        random.shuffle(fold_trainval)

        cases_not_used = [elem for elem in fold_trainval if elem not in already_used]
        if (len(cases_not_used) < 3):
            print("STOP")
            quit()

        fold_val = cases_not_used[:3]
        fold_train = [elem for elem in fold_trainval if elem not in fold_val]
        already_used += fold_val

        print("TRAIN")
        print(fold_train)
        print("VAL")
        print(fold_val)
        print("TEST")
        print(fold_test)
        
        train_cases, val_cases, test_cases = get_cases(fold_train, fold_val, fold_test)
        folds.append({ # Fold 1
            "train": train_cases,
            "val": val_cases, 
            "test": test_cases
        })

    # Define the output directory and filename
    output_directory = "nnUNet_preprocessed/Dataset033_MMWHSCT/"
    output_filename = "splits_final.json"

    # Create the output directory if it does not exist
    os.makedirs(output_directory, exist_ok=True)

    # Define the full path to the output file
    output_file_path = os.path.join(output_directory, output_filename)

    # Save the dictionary to a JSON file
    with open(output_file_path, 'w') as json_file:
        json.dump(folds, json_file, indent=4)





if __name__ == "__main__":
    main()