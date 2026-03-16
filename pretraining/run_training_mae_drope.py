

import pretraining.config as config
import os
import torch

import wandb
import json
import numpy as np
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

from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
from torch.utils.data import DistributedSampler
from utilities.data_utils import DiffMRIDataset_whole, CollateConcat, load_nifti_affine, reconstruct_3d_image
from models.mae_mixed_drope import MaskedAutoEncoderViT
from models.mae_mixed_drope_conv import MaskedAutoEncoderViT as MaskedAutoEncoderViTConv
from utilities.train_utils import cosine_schedule
import random


wandb.login()

# --- Defaults (may be overridden by external JSON) ---
_SPLIT_FOLDER = config.sorted_dataset_folder
_DATASET_FOLDER = config.data_folder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

splits = ['train', 'val']
list_ids = {}
# Load subject split files once at module import time
for split in splits:
    with open(os.path.join(_SPLIT_FOLDER,split+"_subject_list"), "r") as file:
        list_ids[split] = [line.strip() for line in file if line.strip()]

# WandB sweep template (can be used externally)
sweep_config = {
    'method': 'random',
    'metric': {'name':'tensor_mse', 'goal': 'minimize'}
}

def prepare_config_from_json(path):
    """Load JSON config and return a dict with all necessary runtime values.

    Returns a dict containing:
      - parameters_dict (for wandb / mp.spawn)
      - LR_START, LR_END, WD_START, WD_END
      - alpha init/final
      - TASK, MASK_TYPE, IMG_SIZE, INTERVAL_RECON, NAME_PROJ
    """
    # Build a runtime config dict that will be passed to worker processes
    cfg = {}
    try:
        with open(path, 'r') as jf:
            j = json.load(jf)
    except FileNotFoundError:
        print(f"pretraining_config.json not found at {path}, using defaults")
        j = {}
    except Exception as e:
        print(f"Error reading {path}: {e}. Using defaults")
        j = {}

    # Read top-level values
    TASK = j.get('task', 'DTI_alldirdiv30')
    NAME_PROJ = j.get('name_proj', None)

    # Architecture section contains model/arch related params (was parameters)
    arch = j.get('architecture', {})
    IMG_SIZE = tuple(arch.get('img_size', [192,224,4]))
    PATCH_SIZE = tuple(arch.get('patch_size', [16,16,4]))
    RANDOM_SEL = arch.get('random_sel', 1)
    RANDOM_DIR = arch.get('random_dir', 15)
    MASK_TYPE = arch.get('mask_type', 1)
    CONV_LAYERS = arch.get('convolutional_final_layers', False)

    # Training section contains training-specific params
    training = j.get('training', {})
    INTERVAL_RECON = training.get('interval_recon', 50)
    ALPHA_INIT = training.get('alpha_init', 0.05)
    ALPHA_FINAL = training.get('alpha_final', 0.95)

    # Cosine schedule (LR, weight-decay and alpha endpoints)
    cos_cfg = j.get('cosine_schedule', {})
    LR_START = cos_cfg.get('lr_start', 5e-5)
    LR_END = cos_cfg.get('lr_end', 1e-6)
    WD_START = cos_cfg.get('wd_start', 0.04)
    WD_END = cos_cfg.get('wd_end', 0.4)

    # Build parameters_dict (for wandb / sweep) using architecture + training
    parameters_dict = {
        'dataset_name': arch.get('dataset_name', {'value': TASK}),
        'mask_type': arch.get('mask_type', {'value': MASK_TYPE}),
        'mask_ratio_space': arch.get('mask_ratio_space', {'value': 0.7}),
        'mask_ratio_diff': arch.get('mask_ratio_diff', {'value': 0.5}),
        'num_transf_units': arch.get('num_transf_units', {'value': 10}),
        'dec_num_transf_units': arch.get('dec_num_transf_units', {'value': 4}),
        'emb_size': arch.get('emb_size', {'value': 768}),
        'num_head': arch.get('num_head', {'value': 6}),
        # training-related hits that wandb wants
        'optimizer': training.get('optimizer', {'value': 'adam'}),
        'epochs': training.get('epochs', {'value': 300}),
        'batch_size': training.get('batch_size', {'value': 4}),
        'optimizer': training.get('optimizer', {'value': 'adam'}),
        'epochs': training.get('epochs', {'value': 300}),
        'batch_size': training.get('batch_size', {'value': 4}),
        'interval_recon': training.get('interval_recon', 50),
        'convolutional_final_layers': arch.get('convolutional_final_layers', {'value': CONV_LAYERS})
    }

    if CONV_LAYERS:
        parameters_dict['conv_kernel_size'] = arch.get('conv_kernel_size', {'value': 3})
        parameters_dict['num_dec_conv_layers'] = arch.get('num_dec_conv_layers', {'value': 3})

    # Package all runtime values
    cfg.update({
        'parameters_dict': parameters_dict,
        'LR_START': LR_START,
        'LR_END': LR_END,
        'WD_START': WD_START,
        'WD_END': WD_END,
        'ALPHA_INIT': ALPHA_INIT,
        'ALPHA_FINAL': ALPHA_FINAL,
        'TASK': TASK,
        'MASK_TYPE': MASK_TYPE,
        'IMG_SIZE': IMG_SIZE,
        'PATCH_SIZE': PATCH_SIZE,
        'INTERVAL_RECON': INTERVAL_RECON,
        'RANDOM_SEL': RANDOM_SEL,
        'RANDOM_DIR': RANDOM_DIR,
        'NAME_PROJ': NAME_PROJ
    })
    return cfg

def setup(rank, world_size):
    # Initialize distributed process group for multi-GPU training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12365'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()


def train(rank, world_size, run_cfg):
    # Main training entrypoint for each spawned process
    setup(rank, world_size)
    # Unpack runtime config passed from the main process
    params = run_cfg.get('parameters_dict', {})
    TASK = run_cfg.get('TASK')
    IMG_SIZE = run_cfg.get('IMG_SIZE')
    PATCH_SIZE = run_cfg.get('PATCH_SIZE')
    INTERVAL_RECON = run_cfg.get('INTERVAL_RECON')
    RANDOM_SEL = run_cfg.get('RANDOM_SEL')
    RANDOM_DIR = run_cfg.get('RANDOM_DIR')
    LR_START = run_cfg.get('LR_START')
    LR_END = run_cfg.get('LR_END')
    WD_START = run_cfg.get('WD_START')
    WD_END = run_cfg.get('WD_END')
    ALPHA_INIT = run_cfg.get('ALPHA_INIT')
    ALPHA_FINAL = run_cfg.get('ALPHA_FINAL')
    NAME_PROJ = run_cfg.get('NAME_PROJ')

    # Only rank 0 initializes the wandb run and logs
    if rank==0:
        wandb.init(project=NAME_PROJ, config= params)

    # Load training hyperparameters from the combined parameters dict

    batch_size = params['batch_size']['value']
    num_transf_units = params['num_transf_units']['value']
    dec_num_transf_units = params['dec_num_transf_units']['value']
    emb_size = params['emb_size']['value']
    num_head = params['num_head']['value']
    max_epochs = params['epochs']['value']
    mask_space = params['mask_ratio_space']['value']
    mask_diff = params['mask_ratio_diff']['value']
    mask_type = params['mask_type']['value']
    conv_final = params.get('convolutional_final_layers', {}).get('value', False)
    if conv_final:
        conv_kernel_size = params.get('conv_kernel_size', {}).get('value', 3)
        num_dec_conv_layers = params.get('num_dec_conv_layers', {}).get('value', 3)

    # Create cosine schedules (values come from run_cfg)
    lr_schedule = cosine_schedule(start_val=LR_START, end_val=LR_END, total_epochs=max_epochs, warmup_epochs=40)
    wd_schedule = cosine_schedule(start_val=WD_START, end_val=WD_END, total_epochs=max_epochs)
    
    # Create data loaders (DistributedSampler for multi-GPU)
    dataset_train = DiffMRIDataset_whole(folder_path=_DATASET_FOLDER, task = TASK, ids=list_ids['train'], 
                                    format_diff = 'spherical', rand_sel=RANDOM_SEL, rand_dir=RANDOM_DIR, split=True)

    train_sampler = DistributedSampler(dataset_train, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, sampler=train_sampler, pin_memory=True,
                                    collate_fn=CollateConcat())

    dataset_val = DiffMRIDataset_whole(folder_path=_DATASET_FOLDER, task = TASK, ids=list_ids['val'], 
                                    format_diff = 'spherical', include_metrics=False, rand_sel=RANDOM_SEL, rand_dir=RANDOM_DIR, split=True)
    val_sampler = DistributedSampler(dataset_val, num_replicas=world_size, rank=rank, shuffle=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, sampler=val_sampler, pin_memory=True,
                                    collate_fn=CollateConcat())
    
    
    first_batch = next(iter(train_loader))  # Get the first batch
    
    tensor_size = first_batch[1].size()
    print(tensor_size)
    img_size = tensor_size[1]
    num_directions = tensor_size[-1]
    print(f'num subjects: {len(train_loader)}')
    print(f'img size: {img_size}')
    print(f'num directions: {num_directions}')

    # Create model
    if conv_final:
        model = MaskedAutoEncoderViTConv(in_channels=1, img_size=IMG_SIZE, patch_size=PATCH_SIZE, hidden_size=emb_size, mlp_dim=4*emb_size,
        num_layers=num_transf_units, num_heads=num_head, 
        decoder_hidden_size = emb_size, decoder_mlp_dim = emb_size,
        decoder_num_layers = dec_num_transf_units, decoder_num_heads = num_head, proj_type = "conv", pos_embed_type = "sincos",
        decoder_pos_embed_type = "sincos", dropout_rate = 0.1, spatial_dims = 3, qkv_bias = False, save_attn = False, num_directions=num_directions,
        mask_type= mask_type, masking_ratio_space = mask_space, masking_ratio_diff = mask_diff,
        conv_kernel_size = conv_kernel_size, num_conv_layers = num_dec_conv_layers)
    else:
        model = MaskedAutoEncoderViT(in_channels=1, img_size=IMG_SIZE, patch_size=PATCH_SIZE, hidden_size=emb_size, mlp_dim=4*emb_size,
        num_layers=num_transf_units, num_heads=num_head, 
        decoder_hidden_size = emb_size, decoder_mlp_dim = emb_size,
        decoder_num_layers = dec_num_transf_units, decoder_num_heads = num_head, proj_type = "conv", pos_embed_type = "sincos",
        decoder_pos_embed_type = "sincos", dropout_rate = 0.1, spatial_dims = 3, qkv_bias = False, save_attn = False, num_directions=num_directions,
        mask_type= mask_type, masking_ratio_space = mask_space, masking_ratio_diff = mask_diff)
    model = model.to(torch.float)
    model = model.to(rank)

    # Create optimizer (LR and weight-decay updated per-epoch)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=5e-5,  # will be updated per epoch
        betas=(0.9, 0.999),
        weight_decay=0.04  # will also be updated
    )

    # Training loop
    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}/{max_epochs}')
        alpha = ALPHA_INIT + (epoch/max_epochs)*(ALPHA_FINAL-ALPHA_INIT)
        lr = lr_schedule(epoch)
        wd = wd_schedule(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = wd
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss = train_epoch(model, train_loader, optimizer, alpha)
        val_loss = evaluate_epoch_loss(model, val_loader, alpha)

        # Only rank 0 logs and periodically writes reconstructions and checkpoints
        if rank==0:
            wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch+1})

            if (epoch+1)%INTERVAL_RECON == 0:
                # Save reconstructions for two sample IDs and upload to wandb
                rID = '153631'
                im_dir = os.path.join(_DATASET_FOLDER,rID,f'cubes_{TASK}_dipy',f'recon_transf_mixed_{wandb.run.id}')
                img_plot = recon_images(im_dir, model, rID, epoch+1, task=TASK)
                fig, ax = plt.subplots()
                cax = ax.imshow(img_plot, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                wandb.log({f"recon_{rID}": wandb.Image(fig), "epoch": epoch+1})

                rID = '140117'
                im_dir = os.path.join(_DATASET_FOLDER,rID,f'cubes_{TASK}_dipy',f'recon_transf_mixed_{wandb.run.id}')
                img_plot = recon_images(im_dir, model, rID, epoch+1, task=TASK)
                fig, ax = plt.subplots()
                cax = ax.imshow(img_plot, cmap='gray', vmin=0, vmax=1)
                ax.axis('off')
                wandb.log({f"recon_{rID}": wandb.Image(fig), "epoch": epoch+1})
                torch.save(model.state_dict(), os.path.join(wandb.run.dir, f"trained_model_{epoch+1}.pth"))

    if rank == 0:
        wandb.finish()
    cleanup()
    

def reformat_data(positions,rho, theta, phi):

    # Normalize and package spherical coordinates
    positions = positions/300.0 # scale to have features in same ballpark range
    q_pos = torch.stack([rho, theta, phi],2)
    return positions, q_pos

def build_optimizer(network, optimizer, lr, decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=lr, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=lr, weight_decay=decay)
    return optimizer

def weighted_loss(pred, gt, mask, alpha):
    loss = (pred - gt) **2
    loss = loss.mean(dim=-1)
    loss_mask = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    loss_nomask = (loss * (1-mask)).sum() / ((1-mask).sum())
    weighted_loss = (1.0-alpha)*loss_nomask+(alpha)*loss_mask
    return weighted_loss

def train_epoch(model, train_loader, optimizer, alpha):
    model.train()
    train_loss = 0 
    for i, data in enumerate(train_loader, 0):
        print(f'  -- batch {i+1}/{len(train_loader)}')
        __, x, positions, rho, theta, phi = data
        # Prepare batch and compute prediction
        positions, spherical = reformat_data(positions,rho, theta, phi)
        optimizer.zero_grad()
        x = x.to(device)
        positions = positions.to(device)
        spherical = spherical.to(device)
        pred, mask =  model(x, spherical, positions)
    # Backprop and optimizer step
        loss = weighted_loss(pred, model.patchify(x), mask, alpha)
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            loss.backward()
            optimizer.step()
        train_loss += loss.to('cpu').data.numpy()
    return train_loss

def evaluate_epoch_loss(model, val_loader, alpha):
    model.eval()
    valid_loss = 0.0        
    for i, data in enumerate(val_loader, 0):
        __, x, positions, rho, theta, phi = data
        positions, spherical = reformat_data(positions,rho, theta, phi)
        positions = positions.to(device)
        x = x.to(device)
        spherical = spherical.to(device)
        pred, mask =  model(x, spherical, positions)
        # Accumulate validation loss
        loss = weighted_loss(pred, model.patchify(x), mask, alpha)
        valid_loss += loss.to('cpu').data.numpy()
    return valid_loss #/len(val_loader)

def recon_images(_OUT_FOLDER, model, rID, epoch, image_size=(192,224,192), task=None):
    # Reconstruct full-volume images for a given subject id (rID)
    model.eval()
    base_shape = (192,224,192)
    if not os.path.exists(_OUT_FOLDER):
        os.mkdir(_OUT_FOLDER)
    dataset_l = DiffMRIDataset_whole(folder_path=_DATASET_FOLDER, task = task, ids=[rID], 
                               format_diff = 'spherical', include_metrics=True, split=True)
    data_loader = DataLoader(dataset_l, batch_size=1, shuffle=False, 
                                collate_fn=CollateConcat()) 
    base_file = os.path.join(_DATASET_FOLDER,rID,'dti_mni_dipy','dipy_fa.nii.gz')
    __, affine = load_nifti_affine(base_file)
    mini_batch_size = 2
    for i, data in enumerate(data_loader, 0):
        __, x, positions, rho, theta, phi, __ = data
        positions_scaled, spherical = reformat_data(positions,rho, theta, phi)
        dataset = TensorDataset(x,positions_scaled,spherical)
        batch_loader = DataLoader(dataset, batch_size=mini_batch_size, shuffle=False)
        
        pred_tensors_list = []
        mask_list = []
        for j, batch in enumerate(batch_loader,0):
            print(j)
            x_batch, pos_batch, spherical_batch = batch
            x_batch = x_batch.to(device)  # Move to GPU
            pos_batch = pos_batch.to(device)
            spherical_batch = spherical_batch.to(device)
            
            pred, mask =  model(x_batch, spherical_batch, pos_batch)           
            pred_batch = pred.detach().cpu()  # Compute predictions and move to CPU
            mask_batch = mask.detach().cpu()
            pred_tensors_list.append(pred_batch)
            mask_list.append(mask_batch)
        pred_tensors = torch.cat(pred_tensors_list, dim=0)  # (b v c)
        masks = torch.cat(mask_list, dim=0)
        nmasks = repeat(masks, 'b t s -> b t s c', c=np.prod(model.patch_size))
        imgs = model.unpatchify(pred_tensors)
        mask_imgs = model.unpatchify(nmasks)
        
        T = x.shape[-1]
        
        tensor_image = np.zeros(base_shape+(T,))
        mask_image = np.zeros(base_shape+(T,))
        original_image = np.zeros(base_shape+(T,))
    
        pos_flat = positions.reshape((-1,3)).numpy().astype(int)
        vals_flat = imgs.reshape((-1,T)).numpy()
        vals_mask = mask_imgs.reshape((-1,T)).numpy()
        y = model.unpatchify(model.patchify(x))
        vals_orig = y.reshape((-1,T)).numpy()
          
        tensor_image = reconstruct_3d_image(tensor_image, pos_flat, vals_flat)
        mask_image = reconstruct_3d_image(mask_image, pos_flat, vals_mask)
        original_image = reconstruct_3d_image(original_image, pos_flat, vals_orig)

    nib.save(nib.Nifti1Image(tensor_image, affine), _OUT_FOLDER+f"/recon_diff_{epoch}.nii.gz")
    nib.save(nib.Nifti1Image(mask_image, affine), _OUT_FOLDER+"/mask.nii.gz")
    nib.save(nib.Nifti1Image(original_image, affine), _OUT_FOLDER+"/original.nii.gz")

    # find index to plot 
    has_one = mask_image.any(axis=(0, 1, 2))
    first_index = np.argmax(has_one)
    return tensor_image[:,:,74,first_index]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=os.path.join(os.path.dirname(__file__), 'pretraining_config.json'), help='Path to JSON config')
    args = parser.parse_args()

    run_cfg = prepare_config_from_json(args.config)
    world_size = torch.cuda.device_count()
    print(f'Using {world_size} gpus')
    mp.spawn(train, args=(world_size, run_cfg,), nprocs=world_size)