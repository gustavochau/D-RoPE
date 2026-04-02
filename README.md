## Project Overview

This codebase implements D-RoPE and the dMRI transformer as described in [Diffusion MRI Transformer with a Diffusion Space Rotary Positional Embedding (D-RoPE)](https://arxiv.org/abs/2603.25977), as well as code to partially replicate the results. Please notice that because of size and policies of ADNI and HCP datasets, the data cannot be directly reshared. 

## Repository Structure

### Configuration Files

- **`config_adni.py`**: Configuration paths for ADNI dataset (Alzheimer's Disease Neuroimaging Initiative)
- **`config_agdev.py`**: Configuration paths for HCP-Aging and HCP-Development datasets
- **`pretraining/config.py`**: Base configuration paths for pretraining

### Core Directories

#### `models/`
Contains the neural network architectures for the Masked Autoencoder:

- **`mae_mixed_drope_conv.py`**: Main MAE architecture with Diffusion space rotary position embeddings (D-RoPE) for mixed spatial and diffusion direction modeling
- **`mae_mixed_conv.py`**: Main MAE architecture without D-RoPE 
- **`mixedtransformerblock_drope.py`**: Transformer block with D-RoPE for handling both spatial patches and diffusion directions
- **`selfattention_drope.py`**: Self-attention module with D-RoPE

#### `pretraining/`
Self-supervised pretraining code:
- **`run_training_mae_drope.py`**: Main training script for MAE with distributed data parallel (DDP) support
- **`pretraining_config_drope_conv.json`**: Config file with paramters for training with convolutional decoder

##### `pretraining/evaluation/`
Evaluation of pretrained models:
- **`run_reconstructions.py`**: Generate full-volume reconstructions from pretrained models
- **`run_psnr_ssim_evaluation.py`**: Calculate PSNR and SSIM metrics for reconstruction quality
- **`generate_resnet_features.py`**: Extract ResNet features for comparison

#### `downstream/`
Downstream task evaluation (classification and regression):

- **`get_latents.py`**: Extract latent representations from pretrained encoder
- **`aggregate_latents.py`**: Aggregate latent vectors across subjects

##### Task-Specific Subdirectories:
Each subdirectory contains three evaluation strategies:

**`adas/`** - ADAS cognitive score prediction:
- `Finetune_MAE_ADAS.py`: Partial model fine-tuning
- `Frozen_linear_ADAS.ipynb`: Linear probe on frozen features
- `Frozen_mlp_ADAS.ipynb`: MLP head on frozen features

**`age/`** - Age prediction:
- `Finetune_MAE_age.py`: Partial model fine-tuning
- `frozen_linear_age.ipynb`: Linear probe on frozen features
- `frozen_mlp_age.ipynb`: MLP head on frozen features

**`mci/`** - Mild Cognitive Impairment (MCI) classification:
- `Finetune_MAE_MCI.py`: Partial model fine-tuning
- `frozen_linear_MCI.ipynb`: inear probe on frozen features
- `frozen_mlp_MCI.ipynb`: MLP head on frozen features

**`sex/`** - Sex classification:
- `Finetune_MAE_sex.py`: Partial model fine-tuning
- `frozen_linear_sex.ipynb`: inear probe on frozen features
- `frozen_mlp_sex.ipynb`: MLP head on frozen features

#### `utilities/`
Shared utility functions:

- **`data_utils.py`**: Data loading, preprocessing, and coordinate transformations (Cartesian ↔ spherical)
- **`data_utils_downstream.py`**: Dataset classes for downstream tasks with metadata integration
- **`embed_utils.py`**: Position embedding utilities
- **`eval_utils.py`**: Evaluation metrics and helper functions
- **`train_utils.py`**: Training utilities including cosine learning rate schedules

#### `latents/`
Frozen latents obtained with our self-supervised pretrained model. These latents are the same ones used for the downstream tasks results in the main paper.

- **`data_utils.py`**: Data loading, preprocessing, and coordinate transformations (Cartesian ↔ spherical)
- **`data_utils_downstream.py`**: Dataset classes for downstream tasks with metadata integration
- **`embed_utils.py`**: Position embedding utilities
- **`eval_utils.py`**: Evaluation metrics and helper functions
- **`train_utils.py`**: Training utilities including cosine learning rate schedules

#### `splits/`
Train/validation/test splits for different datasets:

##### `splits/adni/`
- Subject lists for ADNI dataset across 5 cross-validation folds
- Files: `train_subject_list_adni_{0-4}`, `val_subject_list_adni_{0-4}`, `test_subject_list_adni_{0-4}`

##### `splits/hcp/`
- Subject lists for HCP datasets (Aging and Development)
- Files: `train_subject_list_{ag|agdev|dev}_{0-4}`, etc.
- `hcpagdev_meta_norm.csv`: Normalized metadata for HCP subjects

#### `metadata/`
- **`adni_metadata.csv`**: behavioral and demographic metadata for ADNI dataset
- **`hcpagdev_metadata.csv`**: behavioral and demographic metadata for ADNI dataset

## Dependencies

Key libraries:
- PyTorch (with CUDA support)
- MONAI (Medical Open Network for AI)
- nibabel (NIfTI file handling)
- einops (tensor operations)
- wandb (experiment tracking)
- pandas, numpy, matplotlib, seaborn
- scikit-learn



