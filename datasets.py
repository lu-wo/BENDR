import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random
import pandas as pd 
import os, fnmatch
import logging
import numpy as np
from memory_profiler import profile

from dn3.utils import min_max_normalize


def load_data_paths(root_dir, file_pattern):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, file_pattern):
            paths.append(os.path.join(dirpath, filename))
    return paths
@profile
class FileSampler:
    def __init__(self, path, window_len, normalize=True, min_val=-1, max_val=1, cap_vals=True) -> None:
        self.path = path 
        self.df = pd.read_csv(self.path)
        self.tensor = torch.tensor(self.df.loc[:, "channel_0":"channel_127"].values, dtype=torch.float32)
        self.max_val = self.tensor.max()
        self.min_val = self.tensor.min()
        
        if cap_vals: # substitute values > 200 with 200
            self.tensor[self.tensor > 200] = 200
        if normalize:
            self.tensor = min_max_normalize(self.tensor, min_val, max_val)
        
        self.cnt = 0
        self.length = len(self.df)
        self.window_len = window_len
        self.max_sample = self.length // self.window_len
        print(f"Init new FileSampler with length {self.length} window length {self.window_len} and {self.max_sample} samples")

    def draw(self, global_min, global_max):
        if self.cnt >= self.max_sample:
            print("Reached max sample")
        #     raise StopIteration
        # randomly sample window length from data
        start = random.randint(0, self.length - self.window_len - 1)
        end = start + self.window_len
        self.cnt += 1
        tensor = self.tensor[start:end, :]
        val = (self.max_val - self.min_val) / (global_max - global_min)
        const = torch.ones((self.window_len, 1)) * val
        tensor = torch.cat((tensor, const), dim=1)
        # check if nan in tensor    
        if torch.isnan(tensor).any():
            logging.info(f"NaN in tensor from {self.path}")
            raise ValueError 
        return tensor

@profile    
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir=None, paths=None, file_pattern="*stream*.csv", window_len=1000, buffer_size=100):
        """
        Args:
            root_dir (string): Directory with all the participants data.
            paths (list): List of paths to participant data files, alternative to root_dir.
            file_pattern (string): Pattern to match files in root_dir.
            window_len (int): Length of window to sample from each file.
            buffer_size (int): Number of files to load into memory.
        """
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.window_len = window_len
        self.participants_data_paths = load_data_paths(self.root_dir, self.file_pattern) if paths is None else paths
        self.nb_files = len(self.participants_data_paths)
        logging.info(f"Found {self.nb_files} files for this dataset.")
        self.buffer_size = min(self.nb_files, buffer_size)
        self.buffer = self._init_buffer()
        self.len = sum(len(self._load_participant_data(path)) for path in self.participants_data_paths) // self.window_len
        self._compute_stats() # init stats 
        logging.info(f"Dataset length: {self.len} samples.")

    def __len__(self):
        return self.len 

    def __getitem__(self, index):
        file_id = random.randint(0, self.buffer_size-1)
        if self.buffer[file_id].cnt >= self.buffer[file_id].max_sample:
            self.buffer[file_id] = self._draw_file_sampler() # replace file/sampler
        return torch.tensor(self.buffer[file_id].draw(self.global_min, self.global_max), dtype=torch.float32)
    
    def _compute_stats(self):
        self.length = 0
        self.global_min = 1
        self.global_max = -1
        for path in self.participants_data_paths: 
            df = self._load_participant_data(path).loc[:, "channel_0":"channel_127"].values 
            self.length += len(df)
            # print(f"min val {df.min().min()} max val {df.max().max()}")
            self.global_min = min(self.global_min, df.min().min())
            self.global_max = max(self.global_max, df.max().max())
        self.length //= self.window_len
    
    def _load_participant_data(self, path):
        df = pd.read_csv(path)
        return df
    
    def _init_buffer(self):
        buffer = [] 
        for _ in range(self.buffer_size):
            buffer.append(self._draw_file_sampler())
        return buffer
    
    def _draw_file_sampler(self):
        while True:
            path = random.choice(self.participants_data_paths)
            sampler = FileSampler(path, self.window_len)
            if sampler.max_sample > 0:
                break
        return sampler

@profile
class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, file_pattern="*stream*.csv", window_len=1000, buffer_size=100, batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.window_len = window_len
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        logging.info(f"Init DataModule with window_len {self.window_len} buffer_size {self.buffer_size} batch_size {self.batch_size}")
        
        """
        Directory structure:
        randomly split data based on participant
        root_dir
        ├── paradim_1
        │   ├── participant_1
        │   │   ├── stream_1.csv
        │   │   ├── stream_2.csv
        │   │   └── ...
        │   ├── participant_2
        ├── paradim_1
        │   ├── ... 
        """
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
            self.train = MultiTaskDataset(paths=self.train_paths, file_pattern=self.file_pattern, window_len=self.window_len, buffer_size=self.buffer_size)
            logging.info(f"Datamodule: Created training dataset with {len(self.train)} samples")
            self.val = MultiTaskDataset(paths=self.val_paths, file_pattern=self.file_pattern, window_len=self.window_len, buffer_size=self.buffer_size)
            logging.info(f"Datamodule: Created validation dataset with {len(self.val)} samples")

        if stage == "test" or stage is None:
            self.test = MultiTaskDataset(paths=self.test_paths, file_pattern=self.file_pattern, window_len=self.window_len, buffer_size=self.buffer_size)
            logging.info(f"Datamodule: Created test dataset with {len(self.test)} samples")

        logging.info(f"Datamodule setup stage {stage} done.")
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size, num_workers=0)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=self.batch_size, num_workers=0)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size, num_workers=0)


class Dummy(torch.utils.data.Dataset):
    def __init__(self, root_dir, file_pattern="*stream*.csv", window_len=1000, buffer_size=20):
        self.window_len = window_len 

    def __len__(self):
        return 32

    def __getitem__(self, index):
        return torch.ones((self.window_len, 128), dtype=torch.float32)
