import os
import config_agdev as config_paths
import config_adni
import glob
import random
import re
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from einops import repeat, rearrange
import nibabel as nib
import pandas as pd
from utilities.data_utils import cartesian_to_spherical_batch


difftensor_col_min = torch.tensor([3.3223e-10, -1.5352e-03, -1.1757e-03,  3.3223e-10, -1.3698e-03, 3.3223e-10],dtype=torch.float32)
difftensor_col_max = torch.tensor([0.0086, 0.0015, 0.0010, 0.0112, 0.0030, 0.0084],dtype=torch.float32)
#_CSV_FILE = os.path.join(config_paths.sorted_dataset_folder,'hcpya-metadata.csv')

def normalize(diff_tensors):
    diff_tensors = (diff_tensors-difftensor_col_min)/(difftensor_col_max-difftensor_col_min)
    return diff_tensors

def get_files_with_y(directory, sel_y=None):
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
    if sel_y is None:
        sel_y = np.random.choice(y_values,1)[0]
    #sel_y=[0]
    #print(sel_y)
    # Filter files with max y
    result = [f for f, y in matched_files if y == sel_y]
    #print(result)
    return result, sel_y

def get_age_and_sex(csv_file_path, subject_id):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Keep only Subject and Gender columns
    #df = df[['Subject', 'Gender']]
    df = df[['src_subject_id','sex','normalize_age']]

    # Map gender to 0 (M) and 1 (F)
    gender_map = {'M': 0, 'F': 1}
    df['Gender_Code'] = df['sex'].map(gender_map)
    
    # Search for the specific subject
    result = df[df['src_subject_id'] == subject_id]
    
    if result.empty:
        print(f"Subject {subject_id} not found.")
        return None
    else:
        return result['Gender_Code'].values[0], result['normalize_age'].values[0]   

def get_age_and_class(csv_file_path, subject_id):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Keep only Subject and Gender columns
    #df = df[['Subject', 'Gender']]
    df = df[['subject_id','entry_age','entry_research_group']]

    class_map = {'CN': 0, 'MCI': 1}
    df['Class_code'] = df['entry_research_group'].map(class_map)

    # Search for the specific subject
    result = df[df['subject_id'] == subject_id]
    
    if result.empty:
        print(f"Subject {subject_id} not found.")
        return None
    else:
        return result['Class_code'].values[0], result['entry_age'].values[0]   

def get_behavioral_score(csv_file_path, subject_id, column_name):
    # Read the CSV file
    df = pd.read_csv(csv_file_path)
    
    # Keep only Subject and Gender columns
    
    df = df[['src_subject_id',column_name]]

    # Search for the specific subject
    result = df[df['src_subject_id'] == subject_id]
    
    if result.empty:
        print(f"Subject {subject_id} not found.")
        return None
    else:
        return result[column_name].values[0]

def get_valid_subjects(csv_path, target_variable):
    df = pd.read_csv(csv_path)
    if "src_subject_id" not in df.columns:
        raise KeyError("CSV must contain a 'src_subject_id' column.")
    if target_variable not in df.columns:
        raise KeyError(f"CSV must contain the target column '{target_variable}'.")
    # consider non-empty strings and non-NaN numerics as valid
    mask = df[target_variable].notna()
    # if numeric, also require finite values
    if np.issubdtype(df[target_variable].dtype, np.number):
        mask &= np.isfinite(df[target_variable].to_numpy())

    valid_subjects = set(df.loc[mask, "src_subject_id"].astype(str))

    return valid_subjects


class DiffMRIDataset_agesex(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', format_diff = 'basic', include_metrics = False, include_wls = False, rand_sel=0, rand_dir = 0, split=False, predef_y=None, behavioral=None):
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
        self.behavioral = behavioral

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
        if self.ids[idx][:3]=="HCA":
            subfolder = 'HCP-Ag-processed'
        elif self.ids[idx][:3]=="HCD":
            subfolder = 'HCP-Dev-processed'
        subject_path = os.path.join(self.folder_path, subfolder, self.ids[idx], 'cubes_'+self.task +'_dipy')

        # if not self.split:
        #     file_pattern = os.path.join(subject_path, "input_sample_*.npz")
        #     files = glob.glob(file_pattern)
        # else:
        #     files = get_files_with_y(subject_path, self.predef_y)
        #     files = [os.path.join(subject_path,x) for x in files]
        
        __, sel_y = get_files_with_y(subject_path, sel_y=self.predef_y)
        #print(sel_y)
        # if self.rand_split:
        #     sel_y = np.random.choice(np.arange(23), size=1, replace=False)[0]
        # else:
        #     sel_y = self.sel_split
        selected_files = [os.path.join(subject_path,f'input_sample_{i}_{sel_y}.npz') for i in range(16,30)] #6,40
        
        #selected_files = [os.path.join(subject_path,f'input_sample_{i}_{sel_y}.npz') for i in range(20,22)] #6,40

        cubes = []
        tensors = []
        positions = []
        sel_directions = []
        bvals = []
        metrics = []

        subject_code = self.ids[idx].split('_')[0]
        sex_code, age = get_age_and_sex(config_paths.metadata_path,subject_code)
        if self.behavioral is not None:
            score = get_behavioral_score(config_paths.behavioral_path,subject_code,self.behavioral)
        # Load and accumulate
        for file in selected_files:
            data = np.load(file)
            cubes.append(data['cubes'])
            positions.append(data['positions'])

            if self.include_metrics:
                cube_number = os.path.basename(file).split('_')[2]
                file_tensors = os.path.join(subject_path,f'metrics_sample_{cube_number}.npz')
                data_extra = np.load(file_tensors)
                metrics.append(data_extra['metrics'])

            #print(data['positions'].shape)
            #tensors.append(data['tensors'])
        sel_directions.append(data['sel_directions'])
        bvals.append(np.array(data['bvals']).reshape(-1,1))
            #metrics.append(data['metrics'])

        cubes = np.concatenate(cubes, axis=-2)
        vecs = np.stack(sel_directions, axis=0)
        vals = np.stack(bvals, axis=0)
        cubes = cubes[np.newaxis,...]
        positions = np.concatenate(positions, axis=-2)
        positions = positions[np.newaxis,...]
        n_vox = cubes.shape[0]

        if self.include_metrics:
            metrics = np.concatenate(metrics, axis=-2)
            metrics = metrics[...,:2] # FA and MD
            metrics = rearrange(metrics, 'h w z d -> d h w z') # Channels first
            metrics = metrics[np.newaxis,...]

        #print(positions.shape)

        # Sample random directions
        if self.rand_dir>0:
            rand_ind = np.random.choice(np.arange(vals.shape[1]), size=self.rand_dir, replace=False)
            cubes = cubes[...,rand_ind]
            vecs = vecs[:,rand_ind,...]
            vals = vals[:,rand_ind,...]
        elif self.rand_dir<0: #negative means fixed order
            num_dir = -self.rand_dir
            cubes = cubes[...,:num_dir]
            vecs = vecs[:,:num_dir,...]
            vals = vals[:,:num_dir,...]
        ids = np.array(subject_code).reshape(1,1) # include subject id for future reference

        if self.format_diff=='basic':
            npy_files = [ids, cubes, positions, vals, vecs]
        elif self.format_diff=='spherical':
            cartesian_dirs = vals*vecs/1000.0 # scale b-values to maintain scale
            rho, theta, phi = cartesian_to_spherical_batch(cartesian_dirs)        
            npy_files = [ids, cubes, positions, rho, theta, phi]

        if self.include_metrics:
            npy_files.append(metrics)
        npy_files.append(np.array(sex_code).reshape(1,1))
        npy_files.append(np.array(age).reshape(1,1))
        if self.behavioral is not None:
            npy_files.append(np.array(score).reshape(1,1))
            
        tensors = [ids]+[torch.tensor(mat, dtype=torch.float32) for mat in npy_files[1:]]
        # for tensor in tensors:
        #     print(tensor.shape)
        tensors = tuple(tensors)
        return tensors



class DiffMRIDataset_agesex_onlymetrics(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', format_diff = 'basic', include_metrics = False, include_wls = False, rand_sel=0, rand_dir = 0, split=False):
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
        if self.ids[idx][:3]=="HCA":
            subfolder = 'HCP-Ag-processed'
        elif self.ids[idx][:3]=="HCD":
            subfolder = 'HCP-Dev-processed'
        subject_path = os.path.join(self.folder_path, subfolder, self.ids[idx], 'undersampled_dti')

       
        
        metrics = []

        subject_code = self.ids[idx].split('_')[0]
        sex_code, age = get_age_and_sex(config_paths.metadata_path,subject_code)
        data = np.load(os.path.join(subject_path,'dti_undersample.npz'))
        fa = data["fa"][...,np.newaxis]
        md = data["md"][...,np.newaxis]
        
        metrics = np.concatenate([fa,md],axis=-1)
        metrics = rearrange(metrics, 'h w z d -> d h w z') # Channels first
        metrics = metrics[np.newaxis,...]

        #print(positions.shape)

        
        ids = np.array(subject_code).reshape(1,1) # include subject id for future reference

        
        npy_files = [ids]
        
        npy_files.append(metrics)
        npy_files.append(np.array(sex_code).reshape(1,1))
        npy_files.append(np.array(age).reshape(1,1))

        #tensors = [torch.tensor(mat, dtype=torch.float32) for mat in npy_files]
        tensors = [ids]+[torch.tensor(mat, dtype=torch.float32) for mat in npy_files[1:]]

        # for tensor in tensors:
        #     print(tensor.shape)
        tensors = tuple(tensors)
        return tensors


class DiffMRIDataset_adni(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', format_diff = 'basic', include_metrics = False, include_wls = False, rand_sel=0, rand_dir = 0, split=False, predef_y=None, behavioral=None):
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
        self.behavioral = behavioral

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
        subject_path = os.path.join(self.folder_path, 'cubes_'+self.task +'_dipy' ,self.ids[idx])

        # if not self.split:
        #     file_pattern = os.path.join(subject_path, "input_sample_*.npz")
        #     files = glob.glob(file_pattern)
        # else:
        #     files = get_files_with_y(subject_path, self.predef_y)
        #     files = [os.path.join(subject_path,x) for x in files]
        
        __, sel_y = get_files_with_y(subject_path, sel_y=self.predef_y)
        #print(sel_y)
        # if self.rand_split:
        #     sel_y = np.random.choice(np.arange(23), size=1, replace=False)[0]
        # else:
        #     sel_y = self.sel_split
        selected_files = [os.path.join(subject_path,f'input_sample_{i}_{sel_y}.npz') for i in range(16,30)] 
        #selected_files = [os.path.join(subject_path,f'input_sample_{i}_{sel_y}.npz') for i in range(20,22)] 
 
        cubes = []
        tensors = []
        positions = []
        sel_directions = []
        bvals = []
        metrics = []

        subject_code = self.ids[idx].split('/')[0]
        group_code, age = get_age_and_class(config_adni.metadata_path,subject_code)
        if self.behavioral is not None:
            score = get_behavioral_score(config_adni.behavioral_path,subject_code,self.behavioral)
        # Load and accumulate
        for file in selected_files:
            data = np.load(file)
            cubes.append(data['cubes'])
            positions.append(data['positions'])

            cube_number = os.path.basename(file).split('_')[2]
#            file_tensors = os.path.join(subject_path,f'metrics_sample_{cube_number}.npz')
#            data_extra = np.load(file_tensors)

            #print(data['positions'].shape)
            #tensors.append(data['tensors'])
        sel_directions.append(data['sel_directions'])
        bvals.append(np.array(data['bvals']).reshape(-1,1))
            #metrics.append(data['metrics'])

        cubes = np.concatenate(cubes, axis=-2)
        vecs = np.stack(sel_directions, axis=0)
        vals = np.stack(bvals, axis=0)
        cubes = cubes[np.newaxis,...]
        positions = np.concatenate(positions, axis=-2)
        positions = positions[np.newaxis,...]
        n_vox = cubes.shape[0]

        if self.include_metrics:
            metrics = np.concatenate(metrics, axis=-2)
            metrics = metrics[...,:2] # FA and MD
            metrics = rearrange(metrics, 'h w z d -> d h w z') # Channels first
            metrics = metrics[np.newaxis,...]

        #print(positions.shape)

        # Sample random directions
        if self.rand_dir>0:
            rand_ind = np.random.choice(np.arange(vals.shape[1]), size=self.rand_dir, replace=False)
            cubes = cubes[...,rand_ind]
            vecs = vecs[:,rand_ind,...]
            vals = vals[:,rand_ind,...]
        elif self.rand_dir<0: #negative means fixed order
            num_dir = -self.rand_dir
            cubes = cubes[...,:num_dir]
            vecs = vecs[:,:num_dir,...]
            vals = vals[:,:num_dir,...]
        ids = np.array(subject_code).reshape(1,1) # include subject id for future reference

        if self.format_diff=='basic':
            npy_files = [ids, cubes, positions, vals, vecs]
        elif self.format_diff=='spherical':
            cartesian_dirs = vals*vecs/1000.0 # scale b-values to maintain scale
            rho, theta, phi = cartesian_to_spherical_batch(cartesian_dirs)        
            npy_files = [ids, cubes, positions, rho, theta, phi]

        if self.include_metrics:
            npy_files.append(metrics)
        npy_files.append(np.array(group_code).reshape(1,1))
        npy_files.append(np.array(age).reshape(1,1))
        if self.behavioral is not None:
            npy_files.append(np.array(score).reshape(1,1))
        tensors = [ids]+[torch.tensor(mat, dtype=torch.float32) for mat in npy_files[1:]]
        # for tensor in tensors:
        #     print(tensor.shape)
        tensors = tuple(tensors)
        return tensors

class DiffMRIDataset_adni_onlymetrics(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', format_diff = 'basic', include_metrics = False, include_wls = False, rand_sel=0, rand_dir = 0, split=False, predef_y=None, behavioral=None):
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
        self.behavioral = behavioral

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
        subject_path = os.path.join(self.folder_path, 'undersampled_dti', self.ids[idx])
        metrics = []
        subject_code = self.ids[idx].split('/')[0]
        group_code, age = get_age_and_class(config_adni.metadata_path,subject_code)
        if self.behavioral is not None:
            score = get_behavioral_score(config_adni.behavioral_path,subject_code,'TOTAL13_z')

        data = np.load(os.path.join(subject_path,'dti_undersample.npz'))
        fa = data["fa"][...,np.newaxis]
        md = data["md"][...,np.newaxis]
        
        metrics = np.concatenate([fa,md],axis=-1)
        metrics = rearrange(metrics, 'h w z d -> d h w z') # Channels first
        metrics = metrics[np.newaxis,...]

        #print(positions.shape)

        
        ids = np.array(subject_code).reshape(1,1) # include subject id for future reference

        
        npy_files = [ids]
        
        npy_files.append(metrics)
        npy_files.append(np.array(group_code).reshape(1,1))
        npy_files.append(np.array(age).reshape(1,1))
        if self.behavioral is not None:
            npy_files.append(np.array(score).reshape(1,1))

        #tensors = [torch.tensor(mat, dtype=torch.float32) for mat in npy_files]
        tensors = [ids]+[torch.tensor(mat, dtype=torch.float32) for mat in npy_files[1:]]

        # for tensor in tensors:
        #     print(tensor.shape)
        tensors = tuple(tensors)

        return tensors





class DiffMRIDataset_Zong(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', rand_sel=0, rand_dir = 0, rand_split=False, rand_voxel = 168):
        """
        Args:
            folder_path (str): Path to the folder containing the .npy files.
            ids (list of str): List of IDs corresponding to the .npy file names (without extension).
        """
        self.folder_path = folder_path
        self.ids = ids
        self.task = task
        self.rand_sel = rand_sel # Don't take all the voxels but a random selection of size rand_sel
        self.rand_dir = rand_dir
        self.rand_split = rand_split
        self.rand_voxel = rand_voxel

    def __len__(self):
        return len(self.ids)

    def reformat_data(self, x, bvals, bvecs):
        num_vox = x.shape[1]
        x = rearrange(x, "b v c -> (b v) c")

        bvals = repeat(bvals, "b c d -> b v c d", v=num_vox)
        bvecs = repeat(bvecs, "b c d -> b v c d", v=num_vox)
        bvals = rearrange(bvals, "b v c d -> (b v) c d ")
        bvecs = rearrange(bvecs, "b v c d -> (b v) c d ")

        x = torch.log(x+1E-8)

        # Process b-values
        # Target categories (1000, 2000, 3000)
        categories = torch.tensor([0, 1000, 2000, 3000])
        # Find the closest category for each value
        indices = torch.argmin(torch.abs(bvals - categories), dim=-1)
        # One-hot encode
        one_hot = torch.nn.functional.one_hot(indices, num_classes=len(categories))
        #print(one_hot.shape)
        
        # print(x.shape)
        # print(one_hot.shape)
        # print(bvecs.shape)
        
        new_x = torch.concatenate([x[...,None], one_hot, bvecs], dim=-1)

        return new_x
    
    def __getitem__(self, idx):
        """
        Loads the .npy file corresponding to the ID at the given index.

        Args:
            idx (int): Index of the item to fetch.

        Returns:
            torch.Tensor: Tensor representation of the .npy data.
        """
        subject_path = os.path.join(self.folder_path, self.ids[idx], 'cubes_'+self.task +'_dipy')
        if self.rand_split:
            sel_y = np.random.choice(np.arange(23), size=1, replace=False)[0]
        else:
            sel_y = 0
        files = [os.path.join(subject_path,f'zong_sample_{i}_{sel_y}.npz') for i in range(16,30)]

        # Sample Random cubes
        if self.rand_sel>0:
            selected_files = random.sample(files, self.rand_sel)
        else:
            selected_files = files

        cubes = []
        b0s = []
        tensors = []
        positions = []
        sel_directions = []
        bvals = []
        metrics = []

        # Load and accumulate
        for file in selected_files:
            data = np.load(file)
            cubes_diff = data['cubes']
            b0 = data['cube_b0']
            #cubes_all = np.concatenate([b0,cubes_diff],-1)
            cubes.append(cubes_diff)
            b0s.append(b0)
            positions.append(data['positions'])
            sel_directions.append(data['sel_directions'])
            bvals.append(np.array(data['bvals']).reshape(-1,1))
            del data
            cube_number = os.path.basename(file).split('_')[2]
            file_tensors = os.path.join(subject_path,f'metrics_sample_{cube_number}.npz')
            data_extra = np.load(file_tensors)
            
            tensors.append(data_extra['tensors'])
            metrics.append(data_extra['metrics'])
        b0_cubes = np.stack(b0s, axis=0)
        cubes = np.stack(cubes, axis=0)
        #print(cubes.shape)
        diff_tensors = np.stack(tensors, axis=0)
        positions = np.stack(positions, axis=0)
        vecs = np.stack(sel_directions, axis=0)
        #print(vecs.shape)
        vals = np.stack(bvals, axis=0)
        #print(vals.shape)
        metrics = np.stack(metrics, axis=0)

    
        # Sample random directions
        if self.rand_dir>0:
            rand_ind = np.random.choice(np.arange(vals.shape[1]), size=self.rand_dir, replace=False)
            cubes = cubes[...,rand_ind]
            vecs = vecs[:,rand_ind,...]
            vals = vals[:,rand_ind,...]
        elif self.rand_dir<0: #negative means fixed order
            num_dir = -self.rand_dir
            cubes = cubes[...,:num_dir]
            vecs = vecs[:,:num_dir,...]
            vals = vals[:,:num_dir,...]
        
        cubes = np.concatenate([b0_cubes,cubes],-1)
        vals = np.concatenate([np.zeros((vals.shape[0],1,1)),vals],1)
        vecs = np.concatenate([np.zeros((vecs.shape[0],1,3)),vecs],1)
        
        # reformat and rand sel voxels   
        cubes = rearrange(cubes, "b h w z c ->b (h w z) c")
        metrics = rearrange(metrics, "b h w z c ->b (h w z) c")
        positions = rearrange(positions, "b h w z c ->b (h w z) c")
        #diff_tensors = rearrange(diff_tensors, "b h w z c ->b (h w z) c")
        total_vox = cubes.shape[1]
        
        if self.rand_voxel>0:
            rand_ind = np.random.choice(np.arange(total_vox), size=self.rand_voxel, replace=False)
            cubes = cubes[:,rand_ind,:]
            metrics = metrics[:,rand_ind,:]
            positions = positions[:,rand_ind,:]
        
        cubes = torch.tensor(cubes, dtype=torch.float32)
        vals = torch.tensor(vals, dtype=torch.float32)
        vecs =  torch.tensor(vecs, dtype=torch.float32)
        metrics = torch.tensor(metrics, dtype=torch.float32)
        metrics = metrics[...,:4] # FA, MD, RD, AD
        metrics[...,1:4] = metrics[...,1:4]*1000
        positions = torch.tensor(positions, dtype=torch.float32)
        subject_code = self.ids[idx].split('_')[0]
        #ids = np.ones((1,1))*int(subject_code[3:10]) # include subject id for future reference
        ids = np.ones_like(cubes[:,:,[0]])*int(subject_code[3:10]) # include subject id for future reference
        ids = torch.tensor(ids, dtype=torch.float32)

        positions = rearrange(positions, "b v c -> (b v) c")
        metrics = rearrange(metrics, "b v c -> (b v) c")
        ids = rearrange(ids, "b v c -> (b v) c")

        
        cubes = self.reformat_data(cubes, vals, vecs)
        tensors = [ids, cubes, positions, metrics]
        tensors = tuple(tensors)
        return tensors

class DiffMRIDataset_DTI(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', format_diff = 'spherical', rand_sel=0, rand_dir = 0, rand_split=False):
        """
        Args:
            folder_path (str): Path to the folder containing the .npy files.
            ids (list of str): List of IDs corresponding to the .npy file names (without extension).
        """
        self.folder_path = folder_path
        self.ids = ids
        self.task = task
        self.rand_sel = rand_sel # Don't take all the voxels but a random selection of size rand_sel
        self.rand_dir = rand_dir
        self.rand_split = rand_split
        self.format_diff = format_diff

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

        if self.rand_split:
            sel_y = np.random.choice(np.arange(23), size=1, replace=False)[0]
        else:
            sel_y = 0
        files = [os.path.join(subject_path,f'input_sample_{i}_{sel_y}.npz') for i in range(16,30)]

        # Sample Random cubes
        if self.rand_sel>0:
            selected_files = random.sample(files, self.rand_sel)
        else:
            selected_files = files

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
            positions.append(data['positions'])
            sel_directions.append(data['sel_directions'])
            bvals.append(np.array(data['bvals']).reshape(-1,1))
            
            cube_number = os.path.basename(file).split('_')[2]
            file_tensors = os.path.join(subject_path,f'metrics_sample_{cube_number}.npz')
            data_extra = np.load(file_tensors)
            
            #tensors.append(data_extra['tensors'])
            metrics.append(data_extra['metrics'])

        cubes = np.stack(cubes, axis=0)
        positions = np.stack(positions, axis=0)
        vecs = np.stack(sel_directions, axis=0)
        vals = np.stack(bvals, axis=0)
        metrics = np.stack(metrics, axis=0)

        metrics = metrics[...,:4] # FA, MD, RD, AD
        metrics[...,1:4] = metrics[...,1:4]*1000

        n_vox = cubes.shape[0]

        # Sample random directions
        if self.rand_dir>0:
            rand_ind = np.random.choice(np.arange(vals.shape[1]), size=self.rand_dir, replace=False)
            cubes = cubes[...,rand_ind]
            vecs = vecs[:,rand_ind,...]
            vals = vals[:,rand_ind,...]
        elif self.rand_dir<0: #negative means fixed order
            num_dir = -self.rand_dir
            cubes = cubes[...,:num_dir]
            vecs = vecs[:,:num_dir,...]
            vals = vals[:,:num_dir,...]
        subject_code = self.ids[idx].split('_')[0]

        #ids = np.ones((1,1))*int(subject_code[3:10]) # include subject id for future reference
        ids = np.ones_like(cubes[:,:,[0]])*int(subject_code[3:10]) # include subject id for future reference
        ids = torch.tensor(ids, dtype=torch.float32)


        if self.format_diff=='basic':
            npy_files = [ids, cubes, positions, vals, vecs, metrics]
        elif self.format_diff=='spherical':
            cartesian_dirs = vals*vecs/1000.0 # scale b-values to maintain scale
            rho, theta, phi = cartesian_to_spherical_batch(cartesian_dirs)        
            npy_files = [ids, cubes, positions, rho, theta, phi, metrics]

        tensors = [torch.tensor(mat, dtype=torch.float32) for mat in npy_files]
        #tensors[-2] = normalize(tensors[-2])
        tensors = tuple(tensors)
        return tensors

class DiffMRIDataset_behavioral_onlymetrics(Dataset):
    def __init__(self, folder_path, ids, task = 'DTI_fixed_dir', format_diff = 'basic', include_metrics = False, include_wls = False, rand_sel=0, rand_dir = 0, split=False, score_name='Executive_Score'):
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
        self.score_name = score_name

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
        if self.ids[idx][:3]=="HCA":
            subfolder = 'HCP-Ag-processed'
        elif self.ids[idx][:3]=="HCD":
            subfolder = 'HCP-Dev-processed'
        subject_path = os.path.join(self.folder_path, subfolder, self.ids[idx], 'undersampled_dti')

       
        
        metrics = []

        subject_code = self.ids[idx].split('_')[0]
        score = get_behavioral_score(config_paths.behavioral_path,subject_code,self.score_name)
        data = np.load(os.path.join(subject_path,'dti_undersample.npz'))
        fa = data["fa"][...,np.newaxis]
        md = data["md"][...,np.newaxis]
        
        metrics = np.concatenate([fa,md],axis=-1)
        metrics = rearrange(metrics, 'h w z d -> d h w z') # Channels first
        metrics = metrics[np.newaxis,...]

        #print(positions.shape)

        
        ids = np.array(subject_code).reshape(1,1) # include subject id for future reference

        
        npy_files = [ids]
        
        npy_files.append(metrics)
        npy_files.append(np.array(score).reshape(1,1))

        #tensors = [torch.tensor(mat, dtype=torch.float32) for mat in npy_files]
        tensors = [ids]+[torch.tensor(mat, dtype=torch.float32) for mat in npy_files[1:]]

        # for tensor in tensors:
        #     print(tensor.shape)
        tensors = tuple(tensors)
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
            return batch[0]  # first is already numpy, rest are tensors

        # Zip the batch (list of tuples) into separate lists
        zipped = list(zip(*batch))

        # First element: numpy arrays → concatenate with np.concatenate
        first = np.concatenate(zipped[0], axis=0)

        # Remaining elements: torch tensors → concatenate with torch.cat
        rest = [torch.cat(tensors, dim=0) for tensors in zipped[1:]]

        return (first, *rest)

