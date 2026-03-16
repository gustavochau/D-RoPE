import numpy as np
import pandas as pd
import nibabel as nib
import pretraining.config as config_paths
import torch
import os
import argparse
#from skimage.metrics import peak_signal_noise_ratio as psnr
#from skimage.metrics import structural_similarity as ssim
#from monai.metrics.fid import FIDMetric
#from scipy.linalg import sqrtm
from pytorch_msssim import ms_ssim

_BVAL_CENTERS = [1000,2000,3000]

def compute_psnr(img1, ref, max_pixel=255.0):
    """
    Compute the PSNR (Peak Signal-to-Noise Ratio) between two images.
    
    Args:
        img1: First image (as a NumPy array).
        img2: Second image (same shape as img1).
        max_pixel: Maximum possible pixel value (e.g., 255 for 8-bit images).
        
    Returns:
        PSNR value in dB.
    """
    mse = np.sum((img1 - ref) ** 2)/np.sum(ref>0)
    #print(mse)
    if mse == 0:
        return float('inf')  # Identical images
    psnr = 10 * np.log10(max_pixel**2 / np.sqrt(mse))
    return psnr

#fid_metric = FIDMetric()

def compute_metrics_per_volume(original_4d: np.ndarray, reconstructed_4d: np.ndarray, data_range=1.0):
    """
    Compute PSNR and SSIM for each 3D volume in a 4D diffusion MRI dataset.
    Returns lists of PSNRs and SSIMs (one per volume).
    """
    num_volumes = original_4d.shape[3]
    psnr_list = []
    ssim_list = []

    for vol_idx in range(num_volumes):
        orig_vol = original_4d[:, :, :, vol_idx]
        recon_vol = reconstructed_4d[:, :, :, vol_idx]

        # Slice-wise metrics (over Z)
        psnr_slices = []
        ssim_slices = []
        for slice_idx in range(orig_vol.shape[2]):
            orig_slice = orig_vol[:, :, slice_idx]
            recon_slice = recon_vol[:, :, slice_idx]
            max_n = np.max(orig_slice)
            if max_n ==0:
                continue
            orig_slice=orig_slice/max_n
            recon_slice=recon_slice/max_n

            psnr_val = compute_psnr(orig_slice, recon_slice, max_pixel=data_range)
            ssim_val = ms_ssim(torch.from_numpy(orig_slice).unsqueeze(0).unsqueeze(0), torch.from_numpy(recon_slice).unsqueeze(0).unsqueeze(0), data_range=data_range, size_average=True)

            if np.isfinite(psnr_val):
                psnr_slices.append(psnr_val)
            ssim_slices.append(ssim_val)

        psnr_list.append(np.mean(psnr_slices))
        ssim_list.append(np.mean(ssim_slices))

    return psnr_list, ssim_list

def evaluate_subjects(subject_list, input_dir, output_dir, name_out_file, data_range=1.0, sel_bval=1000):
    """
    Evaluate PSNR and SSIM for multiple subjects and save per-subject metrics to file.
    subject_list = [(subject_id, original_4d, reconstructed_4d), ...]
    """
    os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(os.path.join(output_dir, name_out_file)):
        print('File already exists')
        return
    summary_records = []
    lower_bound = sel_bval - 100
    upper_bound = sel_bval + 100
    for i, subject_id in enumerate(subject_list):
        print(f'{i+1}/{len(subject_list)}')
        psnr_list, ssim_list = [], []
        for jj in range(0,10):
            print(f'direction {jj}')
            orig_path = os.path.join(input_dir, subject_id, f'original_{jj}.nii.gz')
            if not os.path.exists(orig_path):
                print(f'{orig_path} not found')
                continue
            original =  nib.load(orig_path).get_fdata()
            reconstructed = nib.load(os.path.join(input_dir,subject_id,f'recon_diff_{jj}.nii.gz')).get_fdata()
            bvals = np.load(os.path.join(input_dir,subject_id,f'bvals_{jj}.npy'))
            indices = np.where((bvals > lower_bound) & (bvals < upper_bound))[0]
            p_list, s_list = compute_metrics_per_volume(original[:,:,:,indices], reconstructed[:,:,:,indices], data_range)
            psnr_list += p_list
            ssim_list += s_list
            #fid_list = compute_fid(original_features, recon_features)
            os.makedirs(os.path.join(output_dir, subject_id), exist_ok=True)
            # Save individual metrics
            np.savez_compressed(os.path.join(output_dir, subject_id, f"{subject_id}_metrics_{sel_bval}.npz"),
                                psnr=np.array(psnr_list), ssim=np.array(ssim_list))

            # Summary metrics
        summary_records.append({
            "Subject": subject_id,
            "Mean_PSNR": np.mean(psnr_list),
            "Mean_SSIM": np.mean(ssim_list),
        })

    df = pd.DataFrame(summary_records)
    df.to_csv(os.path.join(output_dir, name_out_file), index=False)
    return df

def main(method: str, sel_split: str = 'val'):
    """Evaluate PSNR/SSIM for a single method.

    Args:
        method: subfolder name under the reconstructions folder containing recon results.
        sel_split: dataset split to evaluate (default 'val').
    """
    main_dir = config_paths.out_recon_folder
    main_out_dir = config_paths.out_recon_folder
    _SPLIT_FOLDER = config_paths.sorted_dataset_folder
    splits = ['val','test']
    list_ids = {}
    for split in splits:
        with open(os.path.join(_SPLIT_FOLDER,split+"_subject_list"), "r") as file:
            list_ids[split] = [line.strip() for line in file if line.strip()]

    for bval in _BVAL_CENTERS:
        print(sel_split)
        subject_list = list_ids[sel_split]
        print(method)
        input_dir = os.path.join(main_dir,method)
        out_dir = os.path.join(main_out_dir,method)
        df = evaluate_subjects(subject_list, input_dir, out_dir, f'summary-{sel_split}-{method}_{bval}.csv', data_range=1.0, sel_bval=bval)
        # optional: inspect df here or return it


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True, help='Method subfolder to evaluate (e.g. 54k5z5ze)')
    parser.add_argument('--split', default='val', help='Which split to evaluate (val or test)')
    args = parser.parse_args()
    main(args.method, args.split)