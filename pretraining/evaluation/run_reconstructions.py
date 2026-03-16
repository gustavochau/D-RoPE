

import pretraining.config as config_paths
import os
import torch
import re
import time

import wandb
import numpy as np
from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import nibabel as nib
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.cuda.amp import autocast, GradScaler
scaler = torch.amp.GradScaler(device='cuda')
import gc

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler


from utilities.data_utils import DiffMRIDataset_whole_b0, CollateConcat, load_nifti_affine, reconstruct_3d_image
from models.mae_mixed import MaskedAutoEncoderViT
from models.mae_mixed_drope import MaskedAutoEncoderViT as MaskedAutoEncoderViT_Rope
from models.mae_mixed_conv import MaskedAutoEncoderViT as MaskedAutoEncoderViT_conv
from models.mae_mixed_drope_conv import MaskedAutoEncoderViT as MaskedAutoEncoderViT_Rope_conv
from utilities.train_utils import cosine_schedule, get_seed_from_id
import json
import argparse

wandb.login()

_MASK_TYPE = 0
_SPLIT_FOLDER = config_paths.sorted_dataset_folder
_DATASET_FOLDER = config_paths.data_folder
_TASK = 'DTI_alldirdiv30'
_NUM_DIRECTIONS = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

splits = ['val','test']
#splits = ['val']
list_ids = {}
for split in splits:
    with open(os.path.join(_SPLIT_FOLDER,split+"_subject_list"), "r") as file:
        list_ids[split] = [line.strip() for line in file if line.strip()]

# Defines wandb sweep

def load_config_json(path):
    """Load JSON config (similar structure to pretraining scripts) and return a dict with parameters and meta.

    Returns a dict with keys: parameters (dict like previous local config), task, img_size, random_sel, random_dir, training
    """
    try:
        with open(path, 'r') as jf:
            j = json.load(jf)
    except Exception:
        j = {}

    TASK = j.get('task', _TASK)
    arch = j.get('architecture', {})
    training = j.get('training', {})

    parameters = {
        'dataset_name': arch.get('dataset_name', {'value': TASK}),
        'optimizer': training.get('optimizer', {'value': 'adam'}),
        'mask_ratio_space': arch.get('mask_ratio_space', {'value': 0.75}),
        'mask_ratio_diff': arch.get('mask_ratio_diff', {'value': 0.5}),
        'mask_type': arch.get('mask_type', {'value': _MASK_TYPE}),
        'num_transf_units': arch.get('num_transf_units', {'value': 10}),
        'dec_num_transf_units': arch.get('dec_num_transf_units', {'value': 4}),
        'emb_size': arch.get('emb_size', {'value': 768}),
        'num_head': arch.get('num_head', {'value': 6}),
        'batch_size': training.get('batch_size', {'value': 1}),
    }

    meta = {
        'task': TASK,
        'img_size': tuple(arch.get('img_size', [192,224,4])),
        'random_sel': arch.get('random_sel', 1),
        'random_dir': arch.get('random_dir', 15),
        'training': training
    }
    return {'parameters': parameters, 'meta': meta}

_IMG_SIZE = (192,224, 4)
#_PATCH_SIZE = (8,8,4)
_PATCH_SIZE = (16,16,4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
print(device)
def reconstruct(model_type, run_folder, list_subjects, config):
    # Loads run configuration
    print(device)

    num_transf_units = config['num_transf_units']['value']
    dec_num_transf_units = config['dec_num_transf_units']['value']
    emb_size = config['emb_size']['value']
    num_head = config['num_head']['value']
    mask_space = config['mask_ratio_space']['value']
    mask_diff = config['mask_ratio_diff']['value']
    mask_type = config['mask_type']['value']
    # optional convolutional decoder settings: config may supply these
    conv_final = config.get('convolutional_final_layer', {}).get('value', None)
    # backwards compatible: also accept flat key if loader provided it
    if conv_final is None:
        conv_final = config.get('CONVOLUTIONAL_FINAL_LAYER', False)
    num_dec_conv_layers = config.get('num_dec_conv_layers', {}).get('value', None)
    if num_dec_conv_layers is None:
        num_dec_conv_layers = config.get('NUM_DEC_CONV_LAYERS', 4)
    conv_kernel_size = config.get('conv_kernel_size', {}).get('value', None)
    if conv_kernel_size is None:
        conv_kernel_size = config.get('CONV_KERNEL_SIZE', 3)
    #conv_kernel_size = config['conv_kernel_size']['value']
    #num_dec_conv_layers = config['num_dec_conv_layers']['value']

    run_name = run_folder.split('-')[-1]
    weights_path = os.path.join(config_paths.wandb_runs_folder, run_folder, 'files', 'trained_model_300.pth')

    # Creates model and loss function; choose conv vs non-conv decoder based on config flag
    if model_type == 'normal':
        if conv_final:
            ModelClass = MaskedAutoEncoderViT_conv
        else:
            ModelClass = MaskedAutoEncoderViT
    else:
        if conv_final:
            ModelClass = MaskedAutoEncoderViT_Rope_conv
        else:
            ModelClass = MaskedAutoEncoderViT_Rope
    if conv_final:
        model = ModelClass(in_channels=1, img_size=_IMG_SIZE, patch_size=_PATCH_SIZE, hidden_size=emb_size, mlp_dim=4*emb_size,
                           num_layers=num_transf_units, num_heads=num_head,
                           masking_ratio_space=mask_space, masking_ratio_diff=mask_diff, mask_type=mask_type,
                           decoder_hidden_size=emb_size, decoder_mlp_dim=emb_size,
                           decoder_num_layers=dec_num_transf_units, decoder_num_heads=num_head, proj_type="conv", pos_embed_type="sincos",
                           decoder_pos_embed_type="sincos", dropout_rate=0.0, spatial_dims=3, qkv_bias=False, save_attn=False, num_directions=_NUM_DIRECTIONS,
                           num_dec_conv_layers=num_dec_conv_layers, conv_kernel_size=conv_kernel_size)
    else:
        model = ModelClass(in_channels=1, img_size=_IMG_SIZE, patch_size=_PATCH_SIZE, hidden_size=emb_size, mlp_dim=4*emb_size,
                           num_layers=num_transf_units, num_heads=num_head,
                           masking_ratio_space=mask_space, masking_ratio_diff=mask_diff, mask_type=mask_type,
                           decoder_hidden_size=emb_size, decoder_mlp_dim=emb_size,
                           decoder_num_layers=dec_num_transf_units, decoder_num_heads=num_head, proj_type = "conv", pos_embed_type="sincos",
                           decoder_pos_embed_type="sincos", dropout_rate=0.0, spatial_dims=3, qkv_bias=False, save_attn=False, num_directions=_NUM_DIRECTIONS)
    model = model.to(torch.float)
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model = model.to(device)

    out_model_folder = os.path.join(config_paths.out_recon_folder, run_name)
    if not os.path.exists(out_model_folder):
        os.mkdir(out_model_folder)

    for i, rID in enumerate(list_subjects):
        print(f'Reconstructing subject {rID}, {i+1}/{len(list_subjects)}')

        pattern = re.compile(r'input_sample_(\d+)_(\d+)\.npz$')
        files = os.listdir(os.path.join(_DATASET_FOLDER,rID,'cubes_'+_TASK +'_dipy'))
        matched_files = []
        y_values = []

        for f in files:
            match = pattern.match(f)
            if match:
                x, y = map(int, match.groups())
                matched_files.append((f, y))
                y_values.append(y)

        y_values = list(set(y_values))
        y_values.sort()
        print(y_values)
        im_dir = os.path.join(out_model_folder,rID)
        if not os.path.exists(im_dir):
            os.mkdir(im_dir)
            
        if len(y_values)>4:
            y_values = y_values[:4]

        for j, sel_y in enumerate(y_values):
            print(f"PRocessing direction set {j+1}/{len(y_values)}")
            recon_images(im_dir, model, rID, (192,224,192), sel_y)

    
def reformat_data(positions,rho, theta, phi):
    positions = positions/300.0 # scale to have features in same ballpark range
    q_pos = torch.stack([rho, theta, phi],2)
    return positions, q_pos

def recon_images(_OUT_FOLDER, model, rID, image_size=(192,224,192), sel_y=None):
    #model = model.to('cpu')
    model.eval()
    base_shape = (192,224,192)
    #base_shape = _IMG_SIZE
    if not os.path.exists(_OUT_FOLDER):
        os.mkdir(_OUT_FOLDER)
    if os.path.exists(_OUT_FOLDER+f"/recon_diff_{sel_y}.nii.gz"):
        return
    dataset_l = DiffMRIDataset_whole_b0(folder_path=_DATASET_FOLDER, task = _TASK, ids=[rID], 
                               format_diff = 'spherical', sel_y=sel_y)
    data_loader = DataLoader(dataset_l, batch_size=1, shuffle=False, 
                                collate_fn=CollateConcat()) 
    base_file = os.path.join(_DATASET_FOLDER,rID,'dti_mni_dipy','dipy_fa.nii.gz')
    __, affine = load_nifti_affine(base_file)
    mini_batch_size = 2
    seed = get_seed_from_id(rID)
    with torch.no_grad():
        for i, data in enumerate(data_loader, 0):
            ids, b0, x, positions, vals, vecs, rho, theta, phi = data
            positions_scaled, spherical = reformat_data(positions,rho, theta, phi)
            dataset = TensorDataset(b0, x,positions_scaled,spherical)
            batch_loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=False)
            
            pred_tensors_list = []
            b0_list = []
            mask_list = []
            for j, batch in enumerate(batch_loader,0):
                seed = int(rID)
                #print(f'{j}/{len(batch_loader)}')
                b0_batch, x_batch, pos_batch, spherical_batch = batch
                
                x_batch = x_batch.to(device)  # Move to GPU
                pos_batch = pos_batch.to(device)
                spherical_batch = spherical_batch.to(device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred, mask =  model(x_batch, spherical_batch, pos_batch, seed)           
                pred_batch = pred.detach().cpu()  # Compute predictions and move to CPU
                mask_batch = mask.detach().cpu()
                pred_tensors_list.append(pred_batch)
                mask_list.append(mask_batch)
                b0_list.append(b0_batch)
            pred_tensors = torch.cat(pred_tensors_list, dim=0)  # (b v c)
            b0s = torch.cat(b0_list, dim=0)
            masks = torch.cat(mask_list, dim=0)
            nmasks = repeat(masks, 'b t s -> b t s c', c=np.prod(model.patch_size))
            imgs = model.unpatchify(pred_tensors)
            #imgs = model.unpatchify(model.patchify(x))
            mask_imgs = model.unpatchify(nmasks)
            
            T = x.shape[-1]
            
            b0_image = np.zeros(base_shape+(1,))
            tensor_image = np.zeros(base_shape+(T,))
            mask_image = np.zeros(base_shape+(T,))
            original_image = np.zeros(base_shape+(T,))
        
            pos_flat = positions.reshape((-1,3)).numpy().astype(int)
            vals_flat = imgs.reshape((-1,T)).numpy()
            b0s_flat = b0s.reshape((-1,1)).numpy()
            vals_mask = mask_imgs.reshape((-1,T)).numpy()
            y = model.unpatchify(model.patchify(x))
            vals_orig = y.reshape((-1,T)).numpy()

            #print(pos_flat[0,0])
            b0_image = reconstruct_3d_image(b0_image, pos_flat, b0s_flat)
            tensor_image = reconstruct_3d_image(tensor_image, pos_flat, vals_flat)
            mask_image = reconstruct_3d_image(mask_image, pos_flat, vals_mask)
            original_image = reconstruct_3d_image(original_image, pos_flat, vals_orig)

            try:
                del x_batch, pos_batch, spherical_batch
            except Exception:
                pass
            # let CUDA reuse memory promptly
            torch.cuda.empty_cache()
            # encourage Python to free any remaining references
            gc.collect()

    if sel_y:
        nib.save(nib.Nifti1Image(b0_image, affine), _OUT_FOLDER+f"/b0_{sel_y}.nii.gz")
        nib.save(nib.Nifti1Image(tensor_image, affine), _OUT_FOLDER+f"/recon_diff_{sel_y}.nii.gz")
        nib.save(nib.Nifti1Image(mask_image, affine), _OUT_FOLDER+f"/mask_{sel_y}.nii.gz")
        nib.save(nib.Nifti1Image(original_image, affine), _OUT_FOLDER+f"/original_{sel_y}.nii.gz")
        np.save(_OUT_FOLDER+f'/bvals_{sel_y}.npy', vals[0,:,:])
        np.save(_OUT_FOLDER+f'/bvecs_{sel_y}.npy', vecs[0,:,:])

    else:
        nib.save(nib.Nifti1Image(b0_image, affine), _OUT_FOLDER + "/b0.nii.gz")
        nib.save(nib.Nifti1Image(tensor_image, affine), _OUT_FOLDER + "/recon_diff.nii.gz")
        nib.save(nib.Nifti1Image(mask_image, affine), _OUT_FOLDER + "/mask.nii.gz")
        nib.save(nib.Nifti1Image(original_image, affine), _OUT_FOLDER + "/original.nii.gz")


    # find index to plot 
    has_one = mask_image.any(axis=(0, 1, 2))
    first_index = np.argmax(has_one)
    return tensor_image[:,:,74,first_index]


def main(run_name: str, config_path: str, arch: str = 'normal', split: str = 'val'):
    """Entry point for reconstruction.

    Args:
        run_name: wandb run folder name (e.g. run-YYYYMMDD_HHMMSS-xxxxx)
        config_path: path to JSON config similar to pretraining scripts
        arch: model architecture string ('normal' or 'rope')
        split: dataset split to use ('val' or 'test')
    """
    # load configuration and set module-level task/img size used by reconstruct
    cfg = load_config_json(config_path)
    global _TASK, _IMG_SIZE
    _TASK = cfg['meta']['task']
    _IMG_SIZE = cfg['meta']['img_size']

    # Choose subject list according to requested split
    current_list_ids = list_ids.get(split, list_ids.get('val', []))
    print(f"Processing {run_name} - {arch} (split={split})")
    start = time.time()
    reconstruct(model_type=arch, run_folder=run_name, list_subjects=current_list_ids[:3], config=cfg['parameters'])
    end = time.time()
    print(f'Elapsed time: {end - start} s')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'pretraining', 'pretraining_config.json'), help='Path to config JSON')
    parser.add_argument('--run_name', type=str, required=True, help='wandb run folder name (e.g. run-YYYYMMDD_HHMMSS-xxxxx)')
    parser.add_argument('--arch', type=str, default='normal', help='model architecture string (normal or rope)')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'test'], help='dataset split to use for reconstruction (val or test)')
    args = parser.parse_args()
    main(run_name=args.run_name, config_path=args.config, arch=args.arch, split=args.split)
    