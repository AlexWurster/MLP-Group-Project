import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import pandas as pd

# from src.configs.configs import DataConfigs

from huggingface_hub import HfApi

class EEGDataset(Dataset):
    """EEG dataset."""

    # def __init__(self, dataset_repo_path, split: str):

    def __init__(self, 
                 dataset_repo_path, 
                 split: str,
                 ):
        
        self.dataset_path = '/home/co-chae/MLP-EEG-Challenge-chaeeun/' # '/Users/chaeeunlee/Documents/VSC_workspaces/kaggle_hms_eeg_private/hms-harmful-brain-activity-classification/'
        train_csv = pd.read_csv(os.path.join(self.dataset_path, f'train.csv'))
        # self.filenames = [f for f in os.listdir(dataset_path) if os.path.isfile(os.path.join(dataset_path, f))]
        
        API = HfApi()

        self.local_dir = './dataset_cache' # data_configs.local_data_dir
        API.snapshot_download(repo_id=dataset_repo_path, local_dir = self.local_dir, repo_type='dataset')
        self.split = split

        sub_dirs = {'train': ['training_data/butter_bandpass/', 'training_targets/butter_bandpass/'], 'test': ['testing_data/butter_bandpass/', 'testing_targets/butter_bandpass/']}

        # /Users/chaeeunlee/Documents/VSC_workspaces/kaggle_hms_eeg/dataset_cache/train/training_data/butter_bandpass/6824934.npy
        self.dataset_path = os.path.join(self.local_dir, f"{self.split}/{sub_dirs[self.split][0]}")

        # ['seizure_prob', 'lpd_prob', 'gpd_prob', 'lrda_prob', 'grda_prob', 'other_prob']

        self.filenames = [f for f in os.listdir(self.dataset_path) if os.path.isfile(os.path.join(self.dataset_path, f))]
        # self.filenames = [filename.split('.')[0] for filename in self.filenames]

        # Calculate total votes across all vote columns
        total_votes = train_csv.filter(like='_vote').sum(axis=1)
        # print(f"train_csv.shape = {train_csv.shape}, total_votes.shape = {total_votes.shape}")

        # Iterate through columns and calculate probabilities for vote columns
        for column in train_csv.columns:
            if column.endswith('_vote'):
                prob_column_name = column.replace('_vote', '_prob')  
                train_csv[prob_column_name] = train_csv[column] / total_votes

        self.train_csv = train_csv

        
    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        datapoint = np.load(os.path.join(self.dataset_path, self.filenames[idx])).astype(np.float32).transpose()
        label_row = self.train_csv[self.train_csv.label_id==int(self.filenames[idx].split('.')[0])]
        # print(f"label_row = {label_row.columns}")
        # print(f"datapoint.shape = {datapoint.shape}") # (10000, 16)
        # import pdb; pdb.set_trace()
        label = [label_row['seizure_prob'], label_row['lpd_prob'], label_row['gpd_prob'], label_row['lrda_prob'], label_row['grda_prob'], label_row['other_prob']]
        label = [float(item.iloc[0]) for item in label] # fix

        # print(f"label = {label}")
        label = np.array(label).astype(np.float32)

        return datapoint, label


# # Example usage
# if __name__ == "__main__":
#     dataset = EEGDataset(dataset_path='/path/to/your/eeg/data', transform=normalize)
#     dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

#     for i, batch in enumerate(dataloader):
#         print(i, batch['eeg'].shape)
#         # Add your processing code here
