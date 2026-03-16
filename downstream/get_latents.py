

import config_agdev as config_paths
import os
import torch
import yaml
import wandb
import numpy as np
import pandas as pd
from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import nibabel as nib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from utilities.data_utils import CollateConcat, load_nifti_affine, reconstruct_3d_image
from models.mae_mixed_drope import MaskedAutoEncoderViT
from monai.networks.layers import trunc_normal_

from utilities.train_utils import cosine_schedule
from utilities.data_utils_downstream import DiffMRIDataset_agesex


wandb.login()

_SPLIT_FOLDER = '/server/u/user/sorted_splits_agdev' #config_paths.sorted_dataset_folder
_DATASET_FOLDER = config_paths.data_folder
_RANDOM_DIR = -12
_NUM_DIRECTIONS = 12
_TASK = 'DTI_alldirdiv30'
# _PRETRAIN_CHECK_POINT = 300
_SELECTED_RUN_PATH = 'run_code'
main_out_dir = '/server/group/user/HCP-AgDev-reconstruction'
_NAME_PROJ = f"{_TASK}_rope_latents"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


splits = ['train','val','test']
list_ids = {}
for split in splits:
    with open(os.path.join(_SPLIT_FOLDER,split+"_subject_list_agdev"), "r") as file:
        list_ids[split] = [line.strip() for line in file if line.strip()]
list_all_subjects = list_ids['train']+list_ids['val']+list_ids['test']

# Defines wandb sweep

sweep_config = {
    'method': 'random',
    'metric': {'name':'tensor_mse', 'goal': 'minimize'}
}

parameters_dict = {
    'dataset_name':{
        'value':_TASK
        },
    'batch_size': {
        'value': 1
        },
    'num_transf_units':{
        'value': 10
        },
    'dec_num_transf_units':{
        'value': 4
        },
    'emb_size':{
        'value': 768
        },
    'num_head':{
        'value': 6
        },
    }

sweep_config['parameters'] = parameters_dict
_IMG_SIZE = (192,224, 4)
_PATCH_SIZE = (16,16,4)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '16352'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size, config, pre_point):
    setup(rank, world_size)
    
    if rank==0:
        wandb.init(project=_NAME_PROJ, config=config)
    #config = wandb.config
    batch_size = config['batch_size']['value']
    num_transf_units = config['num_transf_units']['value']
    dec_num_transf_units = config['dec_num_transf_units']['value']
    emb_size = config['emb_size']['value']
    num_head = config['num_head']['value']

    # Creates model and los function
    model = MaskedAutoEncoderViT(in_channels=1, img_size=_IMG_SIZE, patch_size=_PATCH_SIZE, hidden_size=emb_size, mlp_dim=4*emb_size,
    num_layers=num_transf_units, num_heads=num_head, 
    decoder_hidden_size = emb_size, decoder_mlp_dim = emb_size,
    decoder_num_layers = dec_num_transf_units, decoder_num_heads = num_head, proj_type = "conv", pos_embed_type = "sincos",
    decoder_pos_embed_type = "sincos", dropout_rate = 0.1, spatial_dims = 3, qkv_bias = False, save_attn = False, num_directions=_NUM_DIRECTIONS)

    model = model.to(torch.float)
    weights_path = os.path.join(config_paths.wandb_runs_folder, _SELECTED_RUN_PATH, f'files/trained_model_{pre_point}.pth')

    # Load weights
    model.load_state_dict(torch.load(weights_path, weights_only=True))

    # Get rid of decoder
    del model._modules['decoder_blocks']
    model.eval()
    model = model.to(rank)

    # Get representations
    for ss, subject_id in enumerate(list_all_subjects):    
        print(f'Processing subject {subject_id}, {ss}/{len(list_all_subjects)}')

        # Creates loader for training and validation
        dataset_train = DiffMRIDataset_agesex(folder_path=_DATASET_FOLDER, task = _TASK, ids=[subject_id], 
                                        format_diff = 'spherical', rand_sel=0, rand_dir=_RANDOM_DIR, predef_y=0)
        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, 
                                        collate_fn=CollateConcat()) 
        for i, data in enumerate(train_loader, 0):
            __, x, positions, rho, theta, phi, sex_label, age = data
            spherical = reformat_data(rho, theta, phi)
            x = x.to(rank, non_blocking=True)
            spherical = spherical.to(rank, non_blocking=True)
            #age = age.to(rank, non_blocking=True)
            positions = positions.to(rank, non_blocking=True)
            with torch.no_grad():
                latent, cls = model.forward_nomask(x,spherical,positions)
                y = model.decoder_pred(latent)
                model.img_size = (192,224, 56)
                y = model.unpatchify(y)
            
        # Save representations
        if rank == 0: #save data
            method = _SELECTED_RUN_PATH.split('-')[-1]
            output_dir = os.path.join(main_out_dir,method, subject_id)

            os.makedirs(output_dir, exist_ok=True)

            np.savez_compressed(os.path.join(output_dir,f"latents_features_{pre_point}_0.npz"),
                                latent=latent.cpu().numpy())
            np.savez_compressed(os.path.join(output_dir,f"cls_{pre_point}_0.npz"),
                                cls=cls.cpu().numpy())
            np.savez_compressed(os.path.join(output_dir,f"latents_features_reshaped_{pre_point}_0.npz"),
                                cls=y.cpu().numpy())

    if rank == 0:
        wandb.finish()
    cleanup()
    
def reformat_data(rho, theta, phi):
    q_pos = torch.stack([rho, theta, phi],2)
    return q_pos

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f'Using {world_size} gpus')
    pre_point=300
    mp.spawn(train, args=(world_size,parameters_dict,pre_point), nprocs=world_size)
    pre_point=200
    mp.spawn(train, args=(world_size,parameters_dict,pre_point), nprocs=world_size)
    pre_point=100
    mp.spawn(train, args=(world_size,parameters_dict,pre_point), nprocs=world_size)

