#!/bin/bash
#SBATCH --job-name=test_CHAOS                       # Job name
#SBATCH --partition=rtx2080ti                       # Partition
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --cpus-per-task=4                           # Number of cores
#SBATCH --time=00:30:00                             # Time limit hrs:min:sec
#SBATCH --output=/projects/multimodal_disentanglement_review/nnUNet/jobs/output/slurm_%j.log   
# Standard output and error log
pwd; hostname; date

# Source bashrc, such that the shell is setup properly
source ~/.bashrc
# Activate conda environment pyenv
source /home/a.eijpe/miniconda3/bin/activate
conda activate myenv
export ITK_NIFTI_SFORM_PERMISSIVE=1

# Load cuda and cudnn (make sure versions match)
# eval `spack load --sh cuda@11.3 cudnn@8.2.0.53-11.3`

# Run your command
cd /projects/multimodal_disentanglement_review/nnUNet
# nnUNetv2_plan_and_preprocess -d 39 -pl nnUNetPlannerResEncL --verify_dataset_integrity
# python create_manual_splits.py 
# rsync -avv --info=progress2 --delete /projects/multimodal_disentanglement_review/nnUNet/nnUNet_preprocessed/Dataset038_T1 $SCRATCH
# export nnUNet_preprocessed=$SCRATCH
# Copy data to scratch before training

# VAL and TEST
nnUNetv2_find_best_configuration 39 -c 2d --disable_ensembling -p nnUNetResEncUNetLPlans

# Test
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset039_T2/imagesTs/fold_0 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_0 -d 39 -c 2d -f 0 -p nnUNetResEncUNetLPlans
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset039_T2/imagesTs/fold_1 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_1 -d 39 -c 2d -f 1 -p nnUNetResEncUNetLPlans
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset039_T2/imagesTs/fold_2 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_2 -d 39 -c 2d -f 2 -p nnUNetResEncUNetLPlans
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset039_T2/imagesTs/fold_3 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_3 -d 39 -c 2d -f 3 -p nnUNetResEncUNetLPlans
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset039_T2/imagesTs/fold_4 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_4 -d 39 -c 2d -f 4 -p nnUNetResEncUNetLPlans

# # Change folder names to test
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_0 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_postprocessed/fold_0 -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_1 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_postprocessed/fold_1 -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_2 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_postprocessed/fold_2 -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_3 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_postprocessed/fold_3 -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test/fold_4 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_postprocessed/fold_4 -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json

# Test NA model on
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset038_T1/imagesTs/fold_0 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_0 -d 39 -c 2d -f 0 -p nnUNetResEncUNetLPlans
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset038_T1/imagesTs/fold_1 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_1 -d 39 -c 2d -f 1 -p nnUNetResEncUNetLPlans
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset038_T1/imagesTs/fold_2 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_2 -d 39 -c 2d -f 2 -p nnUNetResEncUNetLPlans
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset038_T1/imagesTs/fold_3 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_3 -d 39 -c 2d -f 3 -p nnUNetResEncUNetLPlans
# nnUNetv2_predict -i /data/groups/beets-tan/a.eijpe/multimodal_raw/Dataset038_T1/imagesTs/fold_4 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_4 -d 39 -c 2d -f 4 -p nnUNetResEncUNetLPlans

# # # TO apply postprocessing to test FS files
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_0 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA_PP/fold_0/ -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_1 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA_PP/fold_1/ -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_2 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA_PP/fold_2/ -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_3 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA_PP/fold_3/ -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json
# nnUNetv2_apply_postprocessing -i nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA/fold_4 -o nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/test_NA_PP/fold_4/ -pp_pkl_file nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -plans_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/plans.json -dataset_json nnUNet_results/Dataset039_T2/nnUNetTrainer__nnUNetResEncUNetLPlans__2d/crossval_results_folds_0_1_2_3_4/dataset.json