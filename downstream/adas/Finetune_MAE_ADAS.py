import config_adni as config_paths
import os
import torch
import wandb
import numpy as np
import pandas as pd
import gc
from torch import nn
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import nibabel as nib
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler

from models.mae_mixed_drope_conv import MaskedAutoEncoderViT
from monai.networks.layers import trunc_normal_

from utilities.train_utils import cosine_schedule
from utilities.data_utils_downstream import DiffMRIDataset_adni, CollateConcat, get_valid_subjects
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, balanced_accuracy_score, average_precision_score
import copy
import argparse

from torch.cuda.amp import autocast, GradScaler
scaler = torch.amp.GradScaler(device='cuda')



wandb.login()

_SPLIT_FOLDER = config_paths.sorted_dataset_folder
_DATASET_FOLDER = config_paths.data_folder
_RANDOM_DIR = -12
_TASK = 'DTI_alldirdiv30'
_INTERVAL_RECON = 2
_PRETRAIN_CHECK_POINT = 300
_SELECTED_RUN_PATH = 'run-20250806_171752-k8pcq4ai'
_TARGET = 'TOTAL13_z'
_NAME_PROJ = f"{_TASK}_finetuneall_maerope_ADAS_LORA_conv"
_NUM_TRAIN_LAYERS = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
weights_path = os.path.join(config_paths.wandb_runs_folder, _SELECTED_RUN_PATH, f'files/trained_model_{_PRETRAIN_CHECK_POINT}.pth')


# Defines wandb sweep

sweep_config = {
    'method': 'random',
    'metric': {'name':'tensor_mse', 'goal': 'minimize'}
}

parameters_dict = {
    'dataset_name':{
        'value':_TASK
        },
    'optimizer': {
        'value': 'adam'
        },
    'checkpoint_used': {
        'value': os.path.join(_SELECTED_RUN_PATH, f'files/trained_model_{_PRETRAIN_CHECK_POINT}.pth')
        },
    'epochs': {
        'value': 20
    },
    'finetune_strat':{
        'value': 1  
    },
    'batch_size': {
        'value': 6
        },
    'num_transf_units':{
        'value': 10
        },
    'dec_num_transf_units':{
        'value': 3
        },
    'emb_size':{
        'value': 384
        },
    'num_head':{
        'value': 3
        },
    'start_wd':{
        'value': 0.05
        },
    'end_wd':{
        'value': 0.01
        },
    'warmup_epochs':{
        'value': 0
        },
    'start_lr':{
        'value': 5e-5
        },
    'end_lr':{
        'value': 1e-6
        },    
    'lora_rank':{
        'value': 12
        },
    'lora_alpha':{
        'value': 12
        }, 
    'mlp_hidden':{
        'value': 512
        },    
    'conv_kernel_size':{
        'value': 3
    },
    'num_dec_conv_layers':{
        'value': 3
    },
    'num_train_layers':{
        'value':_NUM_TRAIN_LAYERS
    }
    }

sweep_config['parameters'] = parameters_dict
_IMG_SIZE = (192,224, 4)
_PATCH_SIZE = (8,8,4)

# -------------------------
# 1. Devices
# -------------------------
num_gpus = torch.cuda.device_count()
for i in range(num_gpus):
    device_name = torch.cuda.get_device_name(i)
    print(f"Device {i}: {device_name}")

device_backbone = torch.device("cuda:0")
device_lora_head = torch.device("cuda:1")

class ViTWithHead(nn.Module):
    def __init__(self, backbone, emb_size, hidden_size, device_backbone, device_head, num_classes=1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.BatchNorm1d(emb_size),
            nn.Linear(emb_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, num_classes)
        )
        # Copy last block
        self.lastblock = copy.deepcopy(self.backbone.blocks[-1])
        # Replace last block of original backbone. Effectively we are separating the last block
        self.backbone.blocks[-1] = nn.Identity()
        self.backbone.blocks = nn.ModuleList(list(self.backbone.blocks[:-1]))

        self.head.apply(self.init_trunc_normal)

        # Ensure inputs are on the backbone's device (GPU 0)
        self.device_backbone = device_backbone
        self.device_head = device_head

        # Move each component to its designated device
        self.backbone.to(self.device_backbone)
        self.lastblock.to(self.device_head)
        self.ln = nn.LayerNorm(emb_size).to(self.device_head)
        self.head.to(self.device_backbone)

    def init_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
        
    def forward(self, x, spherical, positions):
        # Ensure inputs are on the backbone's device (GPU 0)
        x = x.to(self.device_backbone)
        spherical = spherical.to(self.device_backbone)
        positions = positions.to(self.device_backbone)

        # The forward pass runs on GPU 0
        x, cls = self.backbone.forward_nomask(x, spherical, positions)
        
        # Final block
        b,t,s,_ = x.shape
        spherical_expanded = repeat(spherical.to(self.device_head), 'b t d -> b t s d', s=s)
        spherical_expanded = rearrange(spherical_expanded, 'b t s d -> (b s) t d')
        dist_t = self.backbone.compute_distances(spherical_expanded, spherical_expanded)
        components_rope = self.backbone.gen_arguments(dist_t)
        x = rearrange(x,'b t s d -> b (s t) d')
        x = torch.cat((cls, x), dim=1)
        x = self.lastblock(
            x=x.to(self.device_head), 
            b=b, 
            t=t, 
            s=s, 
            components=components_rope
        )        
        x = self.ln(x)
        cls = x[:,0,:].unsqueeze(1)
        cls = cls.squeeze(1)
        pred = self.head(cls.to(self.device_backbone))
        return pred

def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"Trainable parameters: {trainable}")
    print(f"Frozen parameters: {frozen}")

def train(config, split_num=0):

    proj_name = f"{_NAME_PROJ}_cv_split{split_num}"
    wandb.init(project=proj_name, config=config)

    # Load split lists using the requested split number. Filenames are expected
    # to follow the pattern: '<split>_subject_list_adni_{split_num}'
    splits = ['train', 'val', 'test']
    list_ids = {}
    for split in splits:
        split_file = os.path.join(_SPLIT_FOLDER, f"{split}_subject_list_adni_{split_num}")
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Expected split file not found: {split_file}")
        with open(split_file, "r") as file:
            list_ids[split] = [line.strip() for line in file if line.strip()]

    # Filter by available behavioral subjects
    valid_subjects = get_valid_subjects(config_paths.behavioral_path,_TARGET)
    for split in list_ids:
        before = len(list_ids[split])
        list_ids[split] = [sid for sid in list_ids[split] if sid.split('/')[0] in valid_subjects]
        after = len(list_ids[split])
        print(f"{split}: kept {after}/{before} subjects with {_TARGET}")

    batch_size = config['batch_size']['value']
    num_transf_units = config['num_transf_units']['value']
    dec_num_transf_units = config['dec_num_transf_units']['value']
    emb_size = config['emb_size']['value']
    num_head = config['num_head']['value']
    max_epochs = config['epochs']['value']
    start_wd = config['start_wd']['value']
    end_wd = config['end_wd']['value']
    warmup_epochs = config['warmup_epochs']['value']
    start_lr = config['start_lr']['value']
    end_lr = config['end_lr']['value']
    lora_rank = config['lora_rank']['value']
    lora_alpha = config['lora_alpha']['value']
    mlp_hidden = config['mlp_hidden']['value']
    conv_kernel_size = config['conv_kernel_size']['value']
    num_dec_conv_layers = config['num_dec_conv_layers']['value']
    
    lr_schedule = cosine_schedule(start_val=start_lr, end_val=end_lr, 
                                  total_epochs=max_epochs, warmup_epochs=warmup_epochs)
    wd_schedule = cosine_schedule(start_val=start_wd, end_val=end_wd, 
                                  total_epochs=max_epochs)
    
    # Creates loader for training and validation
    dataset_train = DiffMRIDataset_adni(folder_path=_DATASET_FOLDER, task = _TASK, ids=list_ids['train'], 
                                    format_diff = 'spherical', rand_sel=0, rand_dir=_RANDOM_DIR, predef_y=0, behavioral=_TARGET)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, 
                                    collate_fn=CollateConcat()) 
    dataset_val = DiffMRIDataset_adni(folder_path=_DATASET_FOLDER, task = _TASK, ids=list_ids['val'], 
                                    format_diff = 'spherical', rand_sel=0, rand_dir=_RANDOM_DIR, predef_y=0, behavioral=_TARGET)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=True, collate_fn=CollateConcat())
    dataset_test = DiffMRIDataset_adni(folder_path=_DATASET_FOLDER, task = _TASK, ids=list_ids['test'], 
                                    format_diff = 'spherical', rand_sel=0, rand_dir=_RANDOM_DIR, predef_y=0, behavioral=_TARGET)
    test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False, collate_fn=CollateConcat())
                       
            
    first_batch = next(iter(train_loader))  # Get the first batch
    for ff in first_batch:
        print(ff.shape)
    tensor_size = first_batch[1].size()
    print(tensor_size)
    img_size = tensor_size[1]
    num_directions = tensor_size[-1]
    print(f'num batches: {len(train_loader)}')
    print(f'img size: {img_size}')
    print(f'num directions: {num_directions}')

    # Creates model and los function
    backbone = MaskedAutoEncoderViT(in_channels=1, img_size=_IMG_SIZE, patch_size=_PATCH_SIZE, hidden_size=emb_size, mlp_dim=4*emb_size,
    num_layers=num_transf_units, num_heads=num_head, 
    decoder_hidden_size = emb_size, decoder_mlp_dim = emb_size,
    decoder_num_layers = dec_num_transf_units, decoder_num_heads = num_head, proj_type = "conv", pos_embed_type = "sincos",
    decoder_pos_embed_type = "sincos", dropout_rate = 0.1, spatial_dims = 3, qkv_bias = False, save_attn = False, num_directions=num_directions, 
    num_dec_conv_layers = num_dec_conv_layers, conv_kernel_size = conv_kernel_size)

    backbone = backbone.to(torch.float)

    # Load weights
    backbone.load_state_dict(torch.load(weights_path, weights_only=True))

    # Freeze all parameters of the backbone BEFORE applying LoRA
    for param in backbone.parameters():
        param.requires_grad = False
    
    del backbone._modules['decoder_blocks']
    del backbone._modules['decoder_conv']

    # Define LoRA configuration for the last block of the backbone
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "out_proj","linear1","linear2"],
        lora_dropout=0.3,
        bias="none",
    )

    # Create the full model with the backbone and the head, and move it to the appropriate devices
    model = ViTWithHead(
        backbone,
        emb_size=emb_size,
        hidden_size=mlp_hidden,
        num_classes=1,
        device_backbone=device_backbone,
        device_head=device_lora_head
    )
    for param in model.lastblock.parameters():
        param.requires_grad = True

    # Apply LoRA to the last block of the backbone
    model.lastblock = get_peft_model(model.lastblock, lora_config)

    model.lastblock.print_trainable_parameters()
    # The criterion's pos_weight tensor should also be on the correct device
    criterion = nn.MSELoss()

    # The optimizer correctly finds only the trainable parameters (LoRA + head),
    # which already live on GPU 1.
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3, betas=(0.9, 0.999), weight_decay=0.05
    )

    best_val = -float("inf")
    best_epoch = -1
    best_ckpt_path = None
    key_val = 'Correlation'

    for epoch in range(max_epochs):
        print(f'Epoch {epoch+1}/{max_epochs}')
        lr = lr_schedule(epoch)
        wd = wd_schedule(epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            param_group['weight_decay'] = wd
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate_epoch_loss(model, train_loader, criterion)
        
        wandb.log({"train_loss": train_loss, "val_loss": val_loss, "epoch": epoch})  
        if (epoch+1)%_INTERVAL_RECON == 0:
            ckpt_path = os.path.join(wandb.run.dir, f"trained_model_{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)            
            metrics, df_all = evaluate_metrics(model,val_loader)
            metrics["epoch"]=epoch
            wandb.log(metrics)  
            csv_out = os.path.join(wandb.run.dir,f"result-tables-{_TASK}-val-adas_{epoch+1}.cvs")
            df_all.to_csv(csv_out, index=False) 

            cur_val = float(metrics[key_val])
            if cur_val > best_val:
                best_val = cur_val
                best_epoch = epoch + 1
                best_ckpt_path = ckpt_path
    
                csv_out = os.path.join(wandb.run.dir,f"result-tables-{_TASK}-val-sex_best.cvs")
                df_all.to_csv(csv_out, index=False) 
                # Save test evaluation with best validation

                print(f"Loading best checkpoint from epoch {best_epoch} with best metric={best_val:.6f}")
                model.eval()
                __, df_all_test = evaluate_metrics(model,test_loader)
                csv_out = os.path.join(wandb.run.dir,f"result-tables-{_TASK}-test-adas.cvs")
                df_all_test.to_csv(csv_out, index=False) 

    metrics, df_all = evaluate_metrics(model,val_loader)   
    wandb.log(metrics)  
    csv_out = os.path.join(wandb.run.dir,f"result-tables-{_TASK}-val-adas.cvs")
    df_all.to_csv(csv_out, index=False) 
    wandb.summary["best_epoch"] = best_epoch
    wandb.summary["best_val"] = best_val

    wandb.finish()

    
def reformat_data(rho, theta, phi):
    q_pos = torch.stack([rho, theta, phi],2)
    return q_pos

def build_optimizer(network, optimizer, lr, decay):
    if optimizer == "sgd":
        optimizer = torch.optim.SGD(network.parameters(),
                              lr=lr, momentum=0.9)
    elif optimizer == "adam":
        optimizer = torch.optim.Adam(network.parameters(),
                               lr=lr, weight_decay=decay)
    return optimizer

def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0
    for i, data in enumerate(train_loader, 0):
        print(f'{i}/{len(train_loader)}')
        __, x, positions, rho, theta, phi, __, __, score = data
        spherical = reformat_data(rho, theta, phi)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):

            out = model.forward(x,spherical,positions)
            loss = criterion(out,score.to(device_backbone))

        # Do backprop and optimizer step
        if ~(torch.isnan(loss) | torch.isinf(loss)):
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        # Accumulate numeric loss (detach first to avoid keeping graph)
        try:
            train_loss += float(loss.detach().cpu().item())
        except Exception:
            # fallback in case loss is unexpected type
            train_loss += float(loss.cpu().item())

        # free references to tensors that may live on GPU now that we've
        # completed the optimizer step for this iteration
    try:
        del x, positions, rho, theta, phi, score, spherical, out, loss
    except Exception:
        pass
    # allow CUDA to reclaim cached memory and encourage python GC
    torch.cuda.empty_cache()
    gc.collect()
    return train_loss

def evaluate_epoch_loss(model, val_loader, criterion):
    print('evaluating epoch')
    model.eval()
    valid_loss = 0
    # Evaluation should not track gradients. Use torch.no_grad() to avoid
    # storing computation graph / gradients which cause OOM during eval.
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            print(f'{i}/{len(val_loader)}')
            __, x, positions, rho, theta, phi, __, __, score = data
            spherical = reformat_data(rho, theta, phi)

            # autocast is OK for eval to keep consistent dtype, but no_grad
            # prevents grad buffers from being allocated.
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                out = model.forward(x, spherical, positions)
                loss = criterion(out, score.to(device_backbone))

            # detach and move to CPU immediately to free GPU memory
            valid_loss += float(loss.detach().cpu().item())

            # free references to tensors that may live on GPU
        try:
            del x, positions, rho, theta, phi, score, spherical, out, loss
        except Exception:
            pass
        # let CUDA reuse memory promptly
        torch.cuda.empty_cache()
        # encourage Python to free any remaining references
        gc.collect()
    return valid_loss

def evaluate_metrics(model, metric_loader):
    print('evaluating metrics')

    model.eval()
    all_preds = []
    all_labels = []
    all_ids = []
    mini_batch_size = 1
    with torch.no_grad():

        for i, data in enumerate(metric_loader, 0):
            ids, x, positions, rho, theta, phi, __, __, score = data  # `labels` is a binary tensor
            spherical = reformat_data(rho, theta, phi)
            preds = model.forward(x,spherical,positions)
            
            preds = preds.detach().cpu().numpy().flatten()
            all_preds.extend(preds)
            all_labels.extend(score.numpy().flatten())
            all_ids.extend(ids.flatten())
        try:
            del x, positions, rho, theta, phi, score, spherical, logits
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    df_all = pd.DataFrame({'ids': all_ids, 'pred': all_preds, 'gt':all_labels})
    corr_coef = np.corrcoef(all_preds, all_labels)[0, 1]
    mse = np.mean((all_preds - all_labels) ** 2)

    return {"Correlation": corr_coef, "MSE": mse}, df_all

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f'Using {world_size} gpus')
    parser = argparse.ArgumentParser(description='Run training with a chosen split number')
    parser.add_argument('--split_num', type=int, default=0, help='Split number to use for loading subject lists')
    args = parser.parse_args()
    train(parameters_dict, split_num=args.split_num)


