#!/bin/bash
#SBATCH --job-name=rval_recon
#SBATCH --output=rval_recon_%A.out
#SBATCH --error=rval_recon_%A.err
#SBATCH --account=<your_account>
#SBATCH --partition=<your_partition>
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00

export PYTHONPATH="${PYTHONPATH}:/server/u/user/transformer_code_clean" # Point this to your code directory copy
source /home/user/.bashrc # I am using conda, change this to your environment setup
conda activate monai # I am using conda, change this to your environment setup
python3 -u run_reconstructions.py --config ../pretraining_config_drope_noconv.json --run_name "run-20250703_105310-d8zivl2x" --arch "rope" --split "val"
