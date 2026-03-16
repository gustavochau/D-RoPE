import os
import glob
import random
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from einops import repeat
import nibabel as nib

difftensor_col_min = torch.tensor([3.3223e-10, -1.5352e-03, -1.1757e-03,  3.3223e-10, -1.3698e-03, 3.3223e-10],dtype=torch.float32)
difftensor_col_max = torch.tensor([0.0086, 0.0015, 0.0010, 0.0112, 0.0030, 0.0084],dtype=torch.float32)

def reconstruct_3d_image(volume, positions, values):
    """
    Reconstructs a 3D image from xyz positions and corresponding values.
    
    Parameters:
    positions (numpy.ndarray): An (N, 3) array of xyz coordinates.
    values (numpy.ndarray): A (N,) array of values corresponding to each xyz coordinate.
    
    Returns:
    numpy.ndarray: A 3D array of fixed size (145, 174, 145) representing the reconstructed image.
    """  
    # Fill volume with values
    volume[positions[:, 0], positions[:, 1], positions[:, 2],:] = values
    return volume

def load_nifti_affine(file_path):
    """
    Load a NIfTI file and return the image data as a numpy array.
    """
    nifti_img = nib.load(file_path)
    return nifti_img.get_fdata(), nifti_img.affine


def normalize(diff_tensors):
    diff_tensors = (diff_tensors-difftensor_col_min)/(difftensor_col_max-difftensor_col_min)
    return diff_tensors

def cartesian_to_spherical_batch(vectors):
    """
    Convert a batch of Cartesian coordinates to spherical coordinates.
    
    Args:
        vectors (numpy.ndarray): A numpy array of shape (B, V, 3) where the last dimension is (x, y, z).
    
    Returns:
        tuple: Three numpy arrays (r, theta, phi) of shape (B, V), representing the spherical coordinates.
    """
    # Compute radius (r)
    r = np.linalg.norm(vectors, axis=-1)
    
    # Avoid division by zero for theta computation
    z = vectors[..., 2]
    theta = np.arccos(np.clip(z / r, -1.0, 1.0))  # Clip to handle numerical precision issues
    
    # Compute azimuthal angle (phi)
    x, y = vectors[..., 0], vectors[..., 1]
    phi = np.arctan2(y, x)
    
    return r, theta, phi

def get_files_with_y(directory='.', predef_y=None):
    pattern = re.compile(r'input_sample_(\d+)_(\d+)\.npz$')
    files = os.listdir(directory)
    matched_files = []
    y_values = []

    for f in files:
        match = pattern.match(f)
        if match:
            x, y = map(int, match.groups())
            matched_files.append((f, y))
            y_values.append(y)

    y_values = list(set(y_values))
    if not matched_files:
        print("No matching files found.")
        return []
    #print(y_values)
    sel_y = np.random.choice(y_values,1)
    #sel_y=[0]
    #print(sel_y)
    # Filter files with max y
    
    if predef_y:
        result = [f for f, y in matched_files if y == predef_y]
    else:
        result = [f for f, y in matched_files if y == sel_y[0]]
    #print(result)
    return result

class DiffMRIDataset_whole(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', format_diff = 'basic', include_metrics = False, include_wls = False, rand_sel=0, rand_dir = 0, split=False, predef_y=None):
        """
        Args:
            folder_path (str): Path to the folder containing the .npy files.
            ids (list of str): List of IDs corresponding to the .npy file names (without extension).
        """
        self.folder_path = folder_path
        self.ids = ids
        self.task = task
        self.format_diff = format_diff
        self.include_metrics = include_metrics
        self.include_wls = include_wls
        self.rand_sel = rand_sel # Don't take all the voxels but a random selection of size rand_sel
        self.rand_dir = rand_dir
        self.split = split
        self.predef_y = predef_y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Loads the .npy file corresponding to the ID at the given index.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            torch.Tensor: Tensor representation of the .npy data.
        """
        #id_folder = os.path.join(self.folder_path, self.ids[idx], 'cubes_'+self.task +'_dipy')
        subject_path = os.path.join(self.folder_path, self.ids[idx], 'cubes_'+self.task +'_dipy')
        #file_pattern = os.path.join(subject_path, "cube_sample_*.npz")
        #files = glob.glob(file_pattern)
        #print(file_pattern)
        # Sample given set of directions
        #files = get_files_with_y(subject_path)
        #print(files)
        #print(files)
        #files = ['cube_sample_18.npz','cube_sample_19.npz']
        

        if not self.split:
            file_pattern = os.path.join(subject_path, "input_sample_*.npz")
            files = glob.glob(file_pattern)
        else:
            files = get_files_with_y(subject_path, self.predef_y)
            files = [os.path.join(subject_path,x) for x in files]

        # Sample Random cubes
        if self.rand_sel>0:
            selected_files = random.sample(files, self.rand_sel)
        else:
            selected_files = files
        #print(selected_files)
        cubes = []
        tensors = []
        positions = []
        sel_directions = []
        bvals = []
        metrics = []

        # Load and accumulate
        for file in selected_files:
            data = np.load(file)
            cubes.append(data['cubes'])
            #tensors.append(data['tensors'])
            positions.append(data['positions'])
            sel_directions.append(data['sel_directions'])
            bvals.append(np.array(data['bvals']).reshape(-1,1))
            #metrics.append(data['metrics'])

        cubes = np.stack(cubes, axis=0)
        #diff_tensors = np.stack(tensors, axis=0)
        positions = np.stack(positions, axis=0)
        vecs = np.stack(sel_directions, axis=0)
        vals = np.stack(bvals, axis=0)
        #metrics = np.stack(metrics, axis=0)

        n_vox = cubes.shape[0]

        # Sample random directions
        if self.rand_dir>0:
            rand_ind = np.random.choice(np.arange(vals.shape[1]), size=self.rand_dir, replace=False)
            cubes = cubes[...,rand_ind]
            vecs = vecs[:,rand_ind,...]
            vals = vals[:,rand_ind,...]
        ids = np.ones_like(cubes[:,:,[0]])*int(self.ids[idx]) # include subject id for future reference

        if self.format_diff=='basic':
            npy_files = [ids, cubes, positions, vals, vecs]
        elif self.format_diff=='spherical':
            cartesian_dirs = vals*vecs/1000.0 # scale b-values to maintain scale
            rho, theta, phi = cartesian_to_spherical_batch(cartesian_dirs)        
            npy_files = [ids, cubes, positions, rho, theta, phi]

        if self.include_metrics:
            #metrics = np.load(os.path.join(id_folder,'metrics.npy'))[mask]
            npy_files.append(metrics)
        
        # if self.include_wls:
        #     metrics = np.load(os.path.join(id_folder,'metrics.npy'))[mask]
        #     wls_pred_tensors = np.load(os.path.join(id_folder,'wls_tensors.npy'))[mask]
        #     wls_pred_metrics = np.load(os.path.join(id_folder,'wls_metrics.npy'))[mask]
        #     npy_files = [ids, positions, diff_tensors, metrics, wls_pred_tensors, wls_pred_metrics]

        # Load each .npy file and convert to tensor
        # if self.rand_sel>0:
        #     rand_ind = np.random.choice(np.arange(ids.shape[0]), size=self.rand_sel, replace=False)
        #     tensors = [torch.tensor(mat[rand_ind], dtype=torch.float32) for mat in npy_files]
        # else:
        tensors = [torch.tensor(mat, dtype=torch.float32) for mat in npy_files]
        #tensors[2] = normalize(tensors[2])
        tensors = tuple(tensors)
        return tensors


class DiffMRIDataset_whole_b0(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', format_diff = 'basic', include_metrics = False, include_wls = False, rand_sel=0, rand_dir = 0, sel_y=0):
        """
        Args:
            folder_path (str): Path to the folder containing the .npy files.
            ids (list of str): List of IDs corresponding to the .npy file names (without extension).
        """
        self.folder_path = folder_path
        self.ids = ids
        self.task = task
        self.format_diff = format_diff
        self.include_metrics = include_metrics
        self.include_wls = include_wls
        self.rand_sel = rand_sel # Don't take all the voxels but a random selection of size rand_sel
        self.rand_dir = rand_dir
        self.sel_y = sel_y

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Loads the .npy file corresponding to the ID at the given index.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            torch.Tensor: Tensor representation of the .npy data.
        """
        #id_folder = os.path.join(self.folder_path, self.ids[idx], 'cubes_'+self.task +'_dipy')
        subject_path = os.path.join(self.folder_path, self.ids[idx], 'cubes_'+self.task +'_dipy')
        #file_pattern = os.path.join(subject_path, "cube_sample_*.npz")
        #files = glob.glob(file_pattern)
        #print(file_pattern)
        # Sample given set of directions
        #files = get_files_with_y(subject_path)
        #print(files)
        #print(files)
        #files = ['cube_sample_18.npz','cube_sample_19.npz']
        

        file_pattern = os.path.join(subject_path, f"input_sample_*_{self.sel_y}.npz")
        files = glob.glob(file_pattern)


        # Sample Random cubes
        if self.rand_sel>0:
            selected_files = random.sample(files, self.rand_sel)
        else:
            selected_files = files
        #print(selected_files)
        cubes = []
        tensors = []
        positions = []
        sel_directions = []
        bvals = []
        metrics = []
        b0s = []

        # Load and accumulate
        for file in selected_files:
            data = np.load(file)
            cubes.append(data['cubes'])

            #tensors.append(data['tensors'])

            positions.append(data['positions'])
            sel_directions.append(data['sel_directions'])
            bvals.append(np.array(data['bvals']).reshape(-1,1))
            b0s.append(data['cube_b0'])
            #metrics.append(data['metrics'])

        #diff_tensors = np.stack(tensors, axis=0)
        cubes = np.stack(cubes, axis=0)
        positions = np.stack(positions, axis=0)
        vecs = np.stack(sel_directions, axis=0)
        vals = np.stack(bvals, axis=0)
        b0s = np.stack(b0s, axis=0)
        #metrics = np.stack(metrics, axis=0)

        n_vox = b0s.shape[0]

        # Sample random directions
        if self.rand_dir>0:
            rand_ind = np.random.choice(np.arange(vals.shape[1]), size=self.rand_dir, replace=False)
            vecs = vecs[:,rand_ind,...]
            vals = vals[:,rand_ind,...]
        ids = np.ones_like(b0s[:,:,[0]])*int(self.ids[idx]) # include subject id for future reference

        if self.format_diff=='basic':
            npy_files = [ids, b0s, cubes, positions, vals, vecs]
        elif self.format_diff=='spherical':
            cartesian_dirs = vals*vecs/1000.0 # scale b-values to maintain scale
            rho, theta, phi = cartesian_to_spherical_batch(cartesian_dirs)        
            npy_files = [ids, b0s, cubes, positions, vals, vecs, rho, theta, phi]


        
        # if self.include_wls:
        #     metrics = np.load(os.path.join(id_folder,'metrics.npy'))[mask]
        #     wls_pred_tensors = np.load(os.path.join(id_folder,'wls_tensors.npy'))[mask]
        #     wls_pred_metrics = np.load(os.path.join(id_folder,'wls_metrics.npy'))[mask]
        #     npy_files = [ids, positions, diff_tensors, metrics, wls_pred_tensors, wls_pred_metrics]

        # Load each .npy file and convert to tensor
        # if self.rand_sel>0:
        #     rand_ind = np.random.choice(np.arange(ids.shape[0]), size=self.rand_sel, replace=False)
        #     tensors = [torch.tensor(mat[rand_ind], dtype=torch.float32) for mat in npy_files]
        # else:
        tensors = [torch.tensor(mat, dtype=torch.float32) for mat in npy_files]
        #tensors[2] = normalize(tensors[2])
        tensors = tuple(tensors)
        return tensors


class DiffMRIDataset(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', format_diff = 'basic', include_metrics = False, include_wls = False, rand_sel=0, rand_dir = 0):
        """
        Args:
            folder_path (str): Path to the folder containing the .npy files.
            ids (list of str): List of IDs corresponding to the .npy file names (without extension).
        """
        self.folder_path = folder_path
        self.ids = ids
        self.task = task
        self.format_diff = format_diff
        self.include_metrics = include_metrics
        self.include_wls = include_wls
        self.rand_sel = rand_sel # Don't take all the voxels but a random selection of size rand_sel
        self.rand_dir = rand_dir

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        Loads the .npy file corresponding to the ID at the given index.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            torch.Tensor: Tensor representation of the .npy data.
        """
        id_folder = os.path.join(self.folder_path, self.ids[idx], 'cubes_'+self.task +'_dipy')

        cubes = np.load(os.path.join(id_folder,'cubes.npy'))
        # Filter out nan, inf and out of range normalized values
        #mask = ~np.any(np.isnan(cubes) | np.isinf(cubes) | (cubes>3), axis=(1, 2, 3, 4))
        cubes[cubes>3]=0
        mask = ~np.any(np.isnan(cubes) | np.isinf(cubes), axis=(1, 2, 3, 4))
        cubes = cubes[mask]
        #cubes = cubes[...,0]
        #cubes = np.expand_dims(cubes, axis=1)
        #a = np.flip(cubes[:,:,:,:,1],1)
        #print(a.shape)
        #print(cubes.shape)
        #cubes[:,:,:,:,1]=cubes[:,:,::-1,:,1]
        #cubes[:,:,:,:,3]=cubes[:,:,::-1,:,3]
        #cubes[:,:,:,:,5]=cubes[:,:,::-1,:,5]
        n_vox = cubes.shape[0]
        vecs = np.load(os.path.join(id_folder,'sel_directions.npy'))
        vals = np.load(os.path.join(id_folder,'bvals.npy')).reshape(-1,1)

        if self.rand_dir>0:
            rand_ind = np.random.choice(np.arange(vals.shape[0]), size=self.rand_dir, replace=False)
            cubes = cubes[...,rand_ind]
            vecs = vecs[rand_ind,...]
            vals = vals[rand_ind,...]

        vecs = repeat(vecs, 'h w ->r h w', r=n_vox)
        vals = repeat(vals, 'h w ->r h w', r=n_vox)
        diff_tensors = np.load(os.path.join(id_folder,'tensors.npy'))[mask]
        #diff_tensors = normalize(diff_tensors)
        positions = np.load(os.path.join(id_folder,'positions.npy'))[mask] # scale
        ids = np.ones_like(cubes[:,:,[0]])*int(self.ids[idx]) # include subject id for future reference

        if self.format_diff=='basic':
            npy_files = [ids, cubes, diff_tensors, positions, vals, vecs]
        elif self.format_diff=='spherical':
            cartesian_dirs = vals*vecs/1000.0 # scale b-values to maintain scale
            rho, theta, phi = cartesian_to_spherical_batch(cartesian_dirs)        
            npy_files = [ids, cubes, diff_tensors, positions, rho, theta, phi]

        if self.include_metrics:
            metrics = np.load(os.path.join(id_folder,'metrics.npy'))[mask]
            npy_files.append(metrics)
        
        if self.include_wls:
            metrics = np.load(os.path.join(id_folder,'metrics.npy'))[mask]
            wls_pred_tensors = np.load(os.path.join(id_folder,'wls_tensors.npy'))[mask]
            wls_pred_metrics = np.load(os.path.join(id_folder,'wls_metrics.npy'))[mask]
            npy_files = [ids, positions, diff_tensors, metrics, wls_pred_tensors, wls_pred_metrics]

        # Load each .npy file and convert to tensor
        if self.rand_sel>0:
            rand_ind = np.random.choice(np.arange(ids.shape[0]), size=self.rand_sel, replace=False)
            tensors = [torch.tensor(mat[rand_ind], dtype=torch.float32) for mat in npy_files]
        else:
            tensors = [torch.tensor(mat, dtype=torch.float32) for mat in npy_files]
        tensors[2] = normalize(tensors[2])
        tensors = tuple(tensors)
        #print(tensors[1].shape)
        return tensors

class CollateConcat:
    def __call__(self, batch):
        """
        Custom collate function to concatenate tensors in the tuple for batch size > 1.
        Args:
            batch (list of tuples): List of tuples of tensors from the dataset.
        Returns:
            Tuple of concatenated tensors.
        """
        batch_size = len(batch)
        if batch_size == 1:
            return batch[0]
        
        # Zip the batch (list of tuples) into separate lists of tensors
        zipped = zip(*batch)
        # Concatenate each tensor list along the first dimension
        concatenated = tuple(torch.cat(tensors, dim=0) for tensors in zipped)
        
        return concatenated