import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import random
import pandas as pd 
import os, fnmatch

from dn3.utils import min_max_normalize


class FileSampler:
    def __init__(self, path, window_len, min_val, max_val) -> None:
        self.path = path 
        self.df = pd.read_csv(self.path)
        self.tensor = min_max_normalize(torch.tensor(self.df.loc[:, "channel_0":"channel_127"].values, dtype=torch.float32))
        self.cnt = 0
        self.length = len(self.df)
        self.window_len = window_len
        self.max_sample = self.length // self.window_len

    def __iter__(self):
        if self.cnt >= self.max_sample:
            raise StopIteration
        # randomly sample window length from data
        start = random.randint(0, self.length - self.window_len - 1)
        end = start + self.window_len
        self.cnt += 1
        return self.df.loc[start:end, "channel_0":"channel_127"].values

    
class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, file_pattern="*stream*.csv", window_len=1000, buffer_size=20):
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.window_len = window_len
        self.buffer_size = buffer_size
        self.participants_data_paths = self._load_data_paths()
        self.buffer = self._init_buffer()
        self._compute_stats() # init stats 

    def __len__(self):
        
        return sum(len(self._load_participant_data(path)) for path in self.participants_data_paths) // self.window_len

    def __getitem__(self, index):
        
        file_id = random.choice(self.buffer_size)
        if self.buffer[file_id].cnt > self.buffer[file_id].max_sample:
            self.buffer[file_id] = self._draw_file_sampler() # replace file/sampler
        return torch.tensor(self.buffer[file_id].__iter__(), dtype=torch.float32)
    
    def _load_data_paths(self):
        paths = []
        for dirpath, dirnames, filenames in os.walk(self.root_dir):
            for filename in fnmatch.filter(filenames, self.file_pattern):
                paths.append(os.path.join(dirpath, filename))
        return paths
    
    def _compute_stats(self):
        self.length = 0
        self.min_val = 1
        self.max_val = -1
        for path in self.participants_data_paths: 
            df = self._load_participant_data(path)
            self.length += len(df)
            self.min_val = min(self.min_val, df.min().min())
            self.max_val = max(self.max_val, df.max().max())
        self.length //= self.window_len
    
    def _load_participant_data(self, path):
        df = pd.read_csv(path)
        return df
    
    def _init_buffer(self):
        buffer = [] 
        for i in range(self.buffer_size):
            buffer.append(self._draw_file_sampler())
        return buffer
    
    def _draw_file_sampler(self, path):
        while True:
            path = random.choice(self.participants_data_paths)
            sampler = FileSampler(path, self.window_len)
            if sampler.max_sample > 0:
                break
        return sampler
    
class Dummy(torch.utils.data.Dataset):
    def __init__(self, root_dir, file_pattern="*stream*.csv", window_len=1000, buffer_size=20):
        self.window_len = window_len
        

    def __len__(self):
        return 32

    def __getitem__(self, index):
        return torch.ones((self.window_len, 128), dtype=torch.float32)


class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, file_pattern="*stream*.csv", window_len=1000, buffer_size=20, batch_size=32):
        super().__init__()
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.window_len = window_len
        self.buffer_size = buffer_size
        self.batch_size = batch_size
    
    def setup(self, stage: str):
        # self.test = MultiTaskDataset(self.root_dir, self.file_pattern, self.window_len, self.buffer_size)
        # self.val = MultiTaskDataset(self.root_dir, self.file_pattern, self.window_len, self.buffer_size)
        # self.train = MultiTaskDataset(self.root_dir, self.file_pattern, self.window_len, self.buffer_size)
        self.test = Dummy(self.root_dir, self.file_pattern, self.window_len, self.buffer_size)
        self.val = Dummy(self.root_dir, self.file_pattern, self.window_len, self.buffer_size)
        self.train = Dummy(self.root_dir, self.file_pattern, self.window_len, self.buffer_size)
        
    def train_dataloader(self):
        return DataLoader(self.train, batch_size= self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size= self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size= self.batch_size)
