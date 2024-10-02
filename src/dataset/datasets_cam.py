"""Script of PyTorch Datasets"""

# Local Imports

# Standard imports
import os
from pathlib import Path

# Third party imports
import numpy as np
import torch
from torch.utils.data import Dataset

class EEGMontageDataset(Dataset):
    """
    Creates a PyTorch Dataset for a given EEG montage Dataset.

    Attributes:
        data_dir: Directory containing the data.
        target_dir: Directory containing the targets.
        data_names: List of names of the data files used for __getitem__ implementation.
    """
    
    def __init__(self, data_dir:Path, target_dir:Path) -> None:
        self.data_dir = Path(data_dir) # /home/co-chae/rds/hpc-work/home/camwheeler/camerons_datasets/mlp_eeg_data/train/data
        self.target_dir = Path(target_dir) # /home/co-chae/rds/hpc-work/home/camwheeler/camerons_datasets/mlp_eeg_data/train/targets
        self.data_names = os.listdir(data_dir)

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.data_names)

    def __getitem__(self, index:int) -> tuple[torch.tensor, torch.tensor]:
        """
        Given an index, returns the data and target tensors.

        Note: 
            Torch.from_numpy() does not accept np object dtypes. 
            np.float64 was selected to preserve floating point precision, I believe
            that the data give to us is float64.

        Args:
            index: Index of the data to return.
        """
        
        use_float = True
        transpose = True
        
        file_name = self.data_names[index] 
        data_path = self.data_dir.joinpath(file_name) 
        target_path = self.target_dir.joinpath(file_name)
        np_data_obj = np.load(data_path, allow_pickle=True).astype(np.float32) if use_float else np.load(data_path, allow_pickle=True).astype(np.float64)
        # assert np_data_obj.shape == (10000, 16), f"np_data_obj must be a numpy array with shape (10000, 16) but it's {np_data_obj.shape}"
        np_target_obj = np.load(target_path, allow_pickle=True).astype(np.float32) if use_float else np.load(target_path, allow_pickle=True).astype(np.float64)
        
        np_data_obj = np_data_obj.transpose() if transpose else np_data_obj
        np_target_obj = np_target_obj.transpose() if transpose else np_target_obj
        
        assert np_data_obj.shape == (16, 10000), f"np_data_obj must be a numpy array with shape (16, 10000) but it's {np_data_obj.shape}"

        
        return torch.from_numpy(np_data_obj), torch.from_numpy(np_target_obj)
