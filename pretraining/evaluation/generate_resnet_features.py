import os
import torch
import time

import nibabel as nib
import numpy as np
import config as config_paths
#from tqdm import tqdm
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import v2
from monai.transforms import Compose, Resize, ScaleIntensity, EnsureType
from einops import rearrange, repeat
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

# Load pretrained ResNet50 and configure
resnet_weights = ResNet50_Weights.IMAGENET1K_V2
resnet_transform = resnet_weights.transforms()

resnet_model = resnet50(weights=resnet_weights)
resnet_model.fc = torch.nn.Identity()  # remove final classification layer
resnet_model.to(device)
resnet_model.eval()
_SPLIT_FOLDER = config_paths.sorted_dataset_folder


def extract_subject_features(img):
    H, W, Z, D = img.shape
    latent_dim = 2048

    # Rearrange to (Z D) slices of shape (H, W)
    slices = rearrange(img, 'h w z d -> (z d) h w')

    pad_left = 16
    pad_right = 16

    # Pad only the last dimension (w), keep v and h unchanged
    # Format: ((before_v, after_v), (before_h, after_h), (before_w, after_w))
    slices = np.pad(slices, ((0, 0), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    slices = torch.tensor(slices)
    slices = repeat(slices, 'v h w -> v c h w', c=3)
    print(slices.shape)
    # preprocessed_slices = []
    # for slice_2d in slices:
    #     slice_2d = pre_monai(slice_2d.astype(np.float32))              # MONAI preprocessing
    #     slice_rgb = np.stack([slice_2d] * 3, axis=0)                   # to 3 channels (C, H, W)
    #     slice_tensor = torch.tensor(slice_rgb)                        # to Tensor
    #     preprocessed_slices.append(slice_tensor)

    #batch = torch.stack(preprocessed_slices).to(device)               # shape: (Z*D, 3, 224, 224)
    # slices = resnet_transform(slices).to(device)                               # apply ResNet transforms

    # with torch.no_grad():
    #     latent_batch = resnet_model(slices)                            # shape: (Z*D, 2048)
    
    # Create a DataLoader for the slices
    slice_dataset = TensorDataset(slices)
    slice_loader = DataLoader(slice_dataset, batch_size=32)

    # Accumulate latent features
    latent_outputs = []

    with torch.no_grad():
        for (batch,) in slice_loader:  # DataLoader returns a tuple
            batch = resnet_transform(batch).to(device)
            latent = resnet_model(batch)  # shape: (B, 2048)
            latent_outputs.append(latent.cpu())  # Store on CPU to save GPU memory

    # Concatenate all latent features back to a single tensor
    latent_batch = torch.cat(latent_outputs, dim=0)  # shape: (Z*D, 2048)
    
    
    print(latent_batch.shape)



    # Reshape to (Z, D, latent_dim) using einops
    features = rearrange(latent_batch.cpu().numpy(), '(z d) f -> z d f', z=Z, d=D)

    return features  # shape: (Z, D, 2048)


def process_subjects(subject_list, input_dir, output_dir,num):
    """
    Evaluate PSNR and SSIM for multiple subjects and save per-subject metrics to file.
    subject_list = [(subject_id, original_4d, reconstructed_4d), ...]
    """
    os.makedirs(output_dir, exist_ok=True)

    for subject_id in subject_list:
        print(f'Processing subject {subject_id}')
        path_img = os.path.join(input_dir,subject_id,f'original_{num}.nii.gz')
        if os.path.exists(path_img):
            original =  nib.load(path_img).get_fdata()
            reconstructed = nib.load(os.path.join(input_dir,subject_id,f'recon_diff_{num}.nii.gz')).get_fdata()

            original_features = extract_subject_features(original)
            recon_features = extract_subject_features(reconstructed)
            # Save individual metrics
            np.savez_compressed(os.path.join(output_dir,subject_id,f"resnet50_features_{num}.npz"),
                                original_features=original_features, recon_features=recon_features)
        else:
            print("Direction set does not exist")
    return 

if __name__ == "__main__":
    #main_dir = config_paths.out_recon_folder#'/local/HCP-YA-reconstruction-all'
    
    #main_out_dir = config_paths.out_recon_folder#'/local/HCP-YA-reconstruction-all/metrics'
    main_dir = '/server/group/user/HCP-YA-reconstruction-all'
    main_out_dir = '/server/group/user/HCP-YA-reconstruction-all'

    splits = ['test']
    list_ids = {}
    for split in splits:
        with open(os.path.join(_SPLIT_FOLDER,split+"_subject_list"), "r") as file:
            list_ids[split] = [line.strip() for line in file if line.strip()]

    dict_pretrained_models=[]
    dict = {}
    dict['run_path'] = 'run-20250626_130002-54k5z5ze'
    dict['architecture'] = 'normal'
    dict_pretrained_models.append(dict)

    for model_dict in dict_pretrained_models:
        print(f'Processing {model_dict['run_path']} - {model_dict['architecture']}')
        method = model_dict['run_path'].split('-')[-1]
        input_dir = os.path.join(main_dir,method)
        out_dir = os.path.join(main_dir,method)
        for i in range(0,9):
            print(f"Processing direction set {i}/9")
            start = time.time()
            #process_subjects(['140117'], input_dir, out_dir, i)
            process_subjects(list_ids['test'], input_dir, out_dir, i)
            end = time.time()
            print(f'Elapsed time: {end - start} s')


        #df = evaluate_subjects(subject_list, input_dir, out_dir, f'summary-{method}.csv', data_range=1.0)
        #print(df)