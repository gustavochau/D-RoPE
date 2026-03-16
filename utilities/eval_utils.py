import numpy as np
import pandas as pd
from dipy.reconst.dti import from_lower_triangular, fractional_anisotropy, mean_diffusivity, axial_diffusivity, radial_diffusivity, TensorModel, decompose_tensor
from einops import rearrange
import nibabel as nib
import torch

difftensor_col_min = np.array([3.3223e-10, -1.5352e-03, -1.1757e-03,  3.3223e-10, -1.3698e-03, 3.3223e-10])
difftensor_col_max = np.array([0.0086, 0.0015, 0.0010, 0.0112, 0.0030, 0.0084])

def unnormalize_gpu(diff_tensors):
    col_min = torch.tensor(difftensor_col_min, device=diff_tensors.device, dtype=torch.float32, requires_grad=False)
    col_max = torch.tensor(difftensor_col_max, device=diff_tensors.device, dtype=torch.float32, requires_grad=False)
    diff_tensors = diff_tensors*(col_max-col_min)+col_min
    return diff_tensors

def unnormalize(diff_tensors):
    diff_tensors = diff_tensors*(difftensor_col_max-difftensor_col_min)+difftensor_col_min
    return diff_tensors

def compute_angles(vectors_a, vectors_b):
    """
    Compute the angles (in degrees) between corresponding 3D vectors in two lists.
    
    Parameters:
        vectors_a (list of lists or np.array): First list of 3D vectors.
        vectors_b (list of lists or np.array): Second list of 3D vectors.
    
    Returns:
        np.array: Array of angles (in degrees) between corresponding vectors.
    """
    # Compute dot products
    dot_products = np.sum(vectors_a * vectors_b, axis=1)

    # Compute magnitudes (norms)
    norms_a = np.linalg.norm(vectors_a, axis=1)
    norms_b = np.linalg.norm(vectors_b, axis=1)

    # Avoid division by zero (small numerical tolerance)
    cos_theta = np.clip(dot_products / (norms_a * norms_b), -1.0, 1.0)

    # Compute angles in degrees
    angles = np.degrees(np.arccos(cos_theta))
    
    return angles

def get_metrics_from_tensor(fsl_tensors):
    # Convert to full diffusion tensors (n x 3 x 3)
    full_tensors = from_lower_triangular(fsl_tensors[:,[0,1,3,2,4,5]]) # Order is inverted because fsl uses upper triangular, dipy uses lower
    # Compute eigenvalues and eigenvectors using dipy's function
    eigvals, eigvecs = decompose_tensor(full_tensors)
    V1, V2, V3 = eigvecs[..., 0], eigvecs[..., 1], eigvecs[..., 2]
    # Compute diffusion metrics
    FA = fractional_anisotropy(eigvals)[...,np.newaxis]
    MD = mean_diffusivity(eigvals)[...,np.newaxis]
    AD = axial_diffusivity(eigvals)[...,np.newaxis]
    RD = radial_diffusivity(eigvals)[...,np.newaxis]
    return np.concatenate([FA,MD,AD,RD,V1,V2,V3],-1)

def evaluate_dti(performance_data, ids, diff_tensor,metrics,pred_tensors, to_flat=True):
    if to_flat:
        ids = rearrange(ids, "b v c -> (b v) c").numpy()
        diff_tensor = rearrange(diff_tensor, "b v c -> (b v) c").numpy()
        metrics = rearrange(metrics, "b v c -> (b v) c").numpy()
        pred_tensors = rearrange(pred_tensors, "b v c -> (b v) c").numpy()
    pred_tensors = unnormalize(pred_tensors)
    diff_tensor = unnormalize(diff_tensor)
    pred_metrics = get_metrics_from_tensor(pred_tensors)
    #print(pred_metrics.shape)
    
    performance_data['id'].append(str(int(ids[0,0])))
    performance_data['tensor_mae'].append(np.mean(1000.0*np.sum(np.abs(diff_tensor-pred_tensors),1)))
    performance_data['FA_mae'].append(np.mean(np.abs(metrics[:,0]-pred_metrics[:,0])))
    performance_data['MD_mae'].append(np.mean(1000.0*np.abs(metrics[:,1]-pred_metrics[:,1])))
    performance_data['RD_mae'].append(np.mean(1000.0*np.abs(metrics[:,2]-pred_metrics[:,2])))
    performance_data['AD_mae'].append(np.mean(1000.0*np.abs(metrics[:,3]-pred_metrics[:,3])))
    angles_pos = compute_angles(metrics[:,4:7], pred_metrics[:,4:7])
    angles_neg = compute_angles(-metrics[:,4:7], pred_metrics[:,4:7])
    angles = np.min(np.stack([angles_pos,angles_neg],1),1)
    performance_data['V1_angle'].append(np.nanmean(angles))
    return performance_data




def load_nifti_affine(file_path):
    """
    Load a NIfTI file and return the image data as a numpy array.
    """
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine

def reconstruct_dti_tensor(upper_tri):
    """
    Reconstructs a full symmetric DTI tensor from upper triangular elements.
    upper_tri: Tensor of shape (..., 6) containing [Dxx, Dxy, Dxz, Dyy, Dyz, Dzz]
    Returns: Tensor of shape (..., 3, 3)
    """
    Sxx, Sxy, Sxz, Syy, Syz, Szz = torch.unbind(upper_tri, dim=-1)
    
    tensor = torch.stack([
        Sxx, Sxy, Sxz,
        Sxy, Syy, Syz,
        Sxz, Syz, Szz
    ], dim=-1).reshape(*upper_tri.shape[:-1], 3, 3)  # Reshape to (..., 3, 3)

    return tensor

def compute_eigenvalues_vectors(S):
    """
    Computes eigenvalues and eigenvectors of the DTI tensor.
    S: Tensor of shape (..., 3, 3)
    Returns: (eigenvalues, eigenvectors)
    """
    eigvals, eigvecs = torch.linalg.eigh(S)  # Returns sorted eigenvalues (smallest to largest)
    masked_eigvals = torch.relu(eigvals) # Zeroes out negative (non-physical) values
    return masked_eigvals, eigvecs

def compute_DTI_metrics(eigvals, eigvecs):
    """
    Computes FA, MD, AD, RD, and V1 from eigenvalues and eigenvectors.
    eigvals: Tensor of shape (..., 3) containing eigenvalues (sorted).
    eigvecs: Tensor of shape (..., 3, 3) containing eigenvectors.
    
    Returns:
        FA: Fractional Anisotropy
        MD: Mean Diffusivity
        AD: Axial Diffusivity (largest eigenvalue)
        RD: Radial Diffusivity (mean of two smallest eigenvalues)
        V1: Principal eigenvector (corresponding to largest eigenvalue)
    """
    # Mean diffusivity
    MD = eigvals.mean(dim=-1, keepdim=True)

    # Axial and Radial Diffusivity
    AD = eigvals[..., -1]  # Largest eigenvalue
    RD = eigvals[..., :-1].mean(dim=-1)  # Mean of the two smallest eigenvalues

    # FA computation
    numerator = torch.sqrt(((eigvals - MD) ** 2).sum(dim=-1))
    denominator = torch.sqrt((eigvals ** 2).sum(dim=-1))
    FA = (1.5 ** 0.5) * (numerator / (denominator + 1e-8))  # Avoid division by zero

    # Principal eigenvector (corresponding to the largest eigenvalue)
    V1 = eigvecs[..., -1]  # Last column of eigvecs (since eigvals are sorted)

    return FA, MD.squeeze(-1), AD, RD, V1

def diffusion_metrics_loss(pred_upper_tri, gt_metrics):
    """
    Computes a differentiable loss between predicted and ground-truth DTI features.
    
    pred_upper_tri: Tensor of shape (..., 6), predicted upper triangular values.
    gt_tensor: Tensor of shape (..., 3, 3), ground truth full tensor.
    weights: Dictionary of weights for each loss term (default: equal weighting).
    
    Returns:
        losses.
    """
    # Reconstruct tensors
    pred_upper_tri = unnormalize_gpu(pred_upper_tri)
    pred_tensor = reconstruct_dti_tensor(pred_upper_tri)

    # Compute eigenvalues & eigenvectors
    pred_eigvals, pred_eigvecs = compute_eigenvalues_vectors(pred_tensor)
    gt_FA = gt_metrics[...,0]
    gt_MD = gt_metrics[...,1]
    gt_RD = gt_metrics[...,2]
    gt_AD = gt_metrics[...,3]
    gt_V1 = gt_metrics[...,4:7]

    # Compute FA, MD, AD, RD, V1
    pred_FA, pred_MD, pred_AD, pred_RD, pred_V1 = compute_DTI_metrics(pred_eigvals, pred_eigvecs)

    # Define loss components
    FA_loss = torch.nn.functional.l1_loss(pred_FA, gt_FA)
    MD_loss = torch.nn.functional.l1_loss(pred_MD, gt_MD)
    AD_loss = torch.nn.functional.l1_loss(pred_AD, gt_AD)
    RD_loss = torch.nn.functional.l1_loss(pred_RD, gt_RD)
    V1_loss = 1 - torch.abs(torch.sum(pred_V1 * gt_V1, dim=-1)).mean()  # Cosine similarity loss

    return FA_loss, MD_loss, AD_loss, RD_loss, V1_loss