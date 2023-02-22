import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random
import pandas as pd 
import os, fnmatch
import logging
import numpy as np

from dn3.utils import min_max_normalize


def load_data_paths(root_dir, file_pattern):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, file_pattern):
            paths.append(os.path.join(dirpath, filename))
    return paths


class VirtualTensorDataset:
    """
    Loads all files into a list of dataframes and samples from them.
    """
    def __init__(self, root_dir=None, paths=None, file_pattern="*stream*.csv", window_len=1000):
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.window_len = window_len
        self.participants_data_paths = load_data_paths(self.root_dir, self.file_pattern) if paths is None else paths
        self.nb_files = len(self.participants_data_paths)
        logging.info(f"Number of files: {self.nb_files}")
        self.dfs = [pd.read_csv(path) for path in self.participants_data_paths]
        self.tensors = [torch.tensor(df.loc[:, "channel_0":"channel_127"].values, dtype=torch.float32) for df in self.dfs]
        self.tensor_lengths = [len(tensor) for tensor in self.tensors]
        self.max_vals = [tensor.max() for tensor in self.tensors]
        self.min_vals = [tensor.min() for tensor in self.tensors]
        self.max_offsets = [int(len(tensor)%self.window_len / 2) for tensor in self.tensors] # max random offset for each tensor's start sample 
        self.leftover = [len(tensor)%self.window_len for tensor in self.tensors] # leftover times after sampling window_len samples from tensor
        self.offsets = [random.randint(0, max_offset) for max_offset in self.max_offsets] # initialize random offsets for each tensor
        self.global_min = min(self.min_vals)
        self.global_max = max(self.max_vals)
        self.prefix_sum_samples = np.cumsum([length // self.window_len for length in self.tensor_lengths]) # prefix sum of each tensor's samples for fast sampling
        self.len = self.prefix_sum_samples[-1]
        logging.info(f"Number of samples in dataset: {self.len}")
        assert self.len == self.prefix_sum_samples[-1], f"len {self.len} != prefix_sum_samples[-1] {self.prefix_sum_samples[-1]}"

        print(f"prefix sum samples {self.prefix_sum_samples}")
        
        # normalize and add constant channel to each tensor 
        for i, tensor in enumerate(self.tensors):
            tensor = min_max_normalize(tensor, self.min_vals[i], self.max_vals[i])
            val = (self.max_vals[i] - self.min_vals[i]) / (self.global_max - self.global_min)
            const = torch.ones((len(tensor), 1)) * val
            # print(f"tensor shape {tensor.shape}")
            self.tensors[i] = torch.cat((tensor, const), dim=1)
            # print(f"tensor shape {self.tensors[i].shape}")

        self.reset_offsets()

    def reset_offsets(self):
        """
        Has to be called after each epoch
        """
        self.offsets = [random.randint(0, max_offset) for max_offset in self.max_offsets]

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        """
        Sample idx from the virtual dataset consisting of many tensors of shape (samples, window_len, 129)
        """
        # find file idx
        file_idx = np.searchsorted(self.prefix_sum_samples, idx, side="right")
        # find sample idx
        sample_idx = idx - self.prefix_sum_samples[file_idx - 1] if file_idx > 0 else idx
        tensor = self.tensors[file_idx][self.offsets[file_idx]:-(self.leftover[file_idx] - self.offsets[file_idx]), :].view(-1, self.window_len, 129)[sample_idx, :, :]

        # check if nan in tensor    
        if torch.isnan(tensor).any():
            logging.info(f"NaN in tensor from {self.participants_data_paths[file_idx]}")
            raise ValueError 
        
        return tensor






    


# class FileSampler:
#     def __init__(self, path, window_len, normalize=True, min_val=-1, max_val=1, cap_vals=True) -> None:
#         self.path = path 
#         self.df = pd.read_csv(self.path)
#         self.tensor = torch.tensor(self.df.loc[:, "channel_0":"channel_127"].values, dtype=torch.float32)
#         self.max_val = self.tensor.max()
#         self.min_val = self.tensor.min()
        
#         if cap_vals: # substitute values > 200 with 200
#             self.tensor[self.tensor > 200] = 200
#         if normalize:
#             self.tensor = min_max_normalize(self.tensor, min_val, max_val)
        
#         self.length = len(self.df)
#         self.window_len = window_len

#     def draw(self, global_min, global_max):
#         # randomly sample window length from data
#         start = random.randint(0, self.length - self.window_len - 1)
#         end = start + self.window_len
#         # cut from self.tensor
#         tensor = self.tensor[start:end, :]
#         # add constant channel
#         val = (self.max_val - self.min_val) / (global_max - global_min)
#         const = torch.ones((self.window_len, 1)) * val
#         tensor = torch.cat((tensor, const), dim=1)
#         # check if nan in tensor    
#         if torch.isnan(tensor).any():
#             logging.info(f"NaN in tensor from {self.path}")
#             raise ValueError 
#         return tensor

    
# class MultiTaskDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir=None, paths=None, file_pattern="*stream*.csv", window_len=1000):
#         """
#         Args:
#             Stores all files in memory and samples from them.
#             root_dir (string): Directory with all the participants data.
#             paths (list): List of paths to participant data files, alternative to root_dir.
#             file_pattern (string): Pattern to match files in root_dir.
#             window_len (int): Length of window to sample from each file.
#         """
#         self.root_dir = root_dir
#         self.file_pattern = file_pattern
#         self.window_len = window_len
#         self.participants_data_paths = load_data_paths(self.root_dir, self.file_pattern) if paths is None else paths
#         self.nb_files = len(self.participants_data_paths)
#         logging.info(f"Number of files: {self.nb_files}")
#         self.file_samplers = [FileSampler(path, window_len) for path in self.participants_data_paths]
#         self.global_min = min([fs.min_val for fs in self.file_samplers])
#         self.global_max = max([fs.max_val for fs in self.file_samplers])
#         self.len = sum([fs.length for fs in self.file_samplers]) // self.window_len

#     def __len__(self):
#         return self.len 

#     def __getitem__(self, index):
#         file_id = random.randint(0, self.nb_files - 1)
#         return self.file_samplers[file_id].draw(self.global_min, self.global_max)
    
#     def _load_participant_data(self, path):
#         df = pd.read_csv(path)
#         return df
    

class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, file_pattern="*stream*.csv", window_len=1000, batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.window_len = window_len
        self.batch_size = batch_size

        #TODO: make this more general
        dirs = [os.path.join(root_dir, o) for o in os.listdir(root_dir) 
                    if os.path.isdir(os.path.join(root_dir,o))]
        dirs = [os.path.join(o, p) for o in dirs for p in os.listdir(o) 
                    if os.path.isdir(os.path.join(o,p))]    
        logging.info(dirs)

        nb_dirs = len(dirs)
        np.random.seed(42)
        nums = np.random.permutation(nb_dirs)
        train_dirs = [dirs[i] for i in nums[:int(0.8*nb_dirs)]]
        val_dirs = [dirs[i] for i in nums[int(0.8*nb_dirs):int(0.9*nb_dirs)]]
        test_dirs = [dirs[i] for i in nums[int(0.9*nb_dirs):]]

        self.train_paths = []
        for d in train_dirs:
            self.train_paths += load_data_paths(d, file_pattern)
        self.val_paths = []
        for d in val_dirs:
            self.val_paths += load_data_paths(d, file_pattern)
        self.test_paths = []
        for d in test_dirs:
            self.test_paths += load_data_paths(d, file_pattern)

        logging.info(f"Found {len(self.train_paths)} training files.")
        logging.info(f"Found {len(self.val_paths)} validation files.")
        logging.info(f"Found {len(self.test_paths)} test files.")

    def prepare_data(self):
        pass
    
    def setup(self, stage: str=None):
        logging.info(f"Setting up datamodule with stage {stage}")
        
        if stage == "fit" or stage is None:
            self.train = VirtualTensorDataset(paths=self.train_paths, file_pattern=self.file_pattern, window_len=self.window_len)
            logging.info(f"Datamodule: Created training dataset with {len(self.train)} samples")
            self.val = VirtualTensorDataset(paths=self.val_paths, file_pattern=self.file_pattern, window_len=self.window_len)
            logging.info(f"Datamodule: Created validation dataset with {len(self.val)} samples")

        if stage == "test" or stage is None:
            self.test = VirtualTensorDataset(paths=self.test_paths, file_pattern=self.file_pattern, window_len=self.window_len)
            logging.info(f"Datamodule: Created test dataset with {len(self.test)} samples")

        logging.info(f"Datamodule setup stage {stage} done.")
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=4)
