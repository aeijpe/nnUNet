#!/bin/bash
#SBATCH --job-name=run_CHAOS                        # Job name
#SBATCH --partition=a6000                           # Partition
#SBATCH --qos=a6000_qos                             # Partition
#SBATCH --gpus-per-task=1                           # Number of gpus per node
#SBATCH --gpus=1                                    # Number of gpus in total
#SBATCH --ntasks=1                                  # Run on a single node
#SBATCH --cpus-per-task=16                          # Number of cores
#SBATCH --time=03:00:00                             # Time limit hrs:min:sec
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

# Train
cd /projects/multimodal_disentanglement_review/nnUNet
rsync -avv --info=progress2 --delete /projects/multimodal_disentanglement_review/nnUNet/nnUNet_preprocessed/Dataset039_T2 $SCRATCH
export nnUNet_preprocessed=$SCRATCH
nnUNetv2_train 39 2d 0 -p nnUNetResEncUNetLPlans --c