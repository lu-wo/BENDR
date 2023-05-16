import torch
from torch.utils.data import DataLoader, Sampler
import pytorch_lightning as pl
import random
import pandas as pd
import os, fnmatch
import logging
import numpy as np
from collections import Counter

from dataset.dataset import create_annotations_num_classes
from typing import Sequence


def min_max_normalize(x: torch.Tensor, low=-1, high=1):
    if len(x.shape) == 2:
        xmin = x.min()
        xmax = x.max()
        if xmax - xmin == 0:
            x = 0
            return x
    elif len(x.shape) == 3:
        xmin = torch.min(torch.min(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        xmax = torch.max(torch.max(x, keepdim=True, dim=1)[0], keepdim=True, dim=-1)[0]
        constant_trials = (xmax - xmin) == 0
        if torch.any(constant_trials):
            # If normalizing multiple trials, stabilize the normalization
            xmax[constant_trials] = xmax[constant_trials] + 1e-6

    x = (x - xmin) / (xmax - xmin)

    # Now all scaled 0 -> 1, remove 0.5 bias
    x -= 0.5
    # Adjust for low/high bias and scale up
    x += (high + low) / 2
    return (high - low) * x

def dataset_summaries(dataset, num_queries):
    max_targets = 0
    labels = []
    no_class = 0
    overshoot = 0
    for i in range(len(dataset)):
        labelled, _, targets = dataset[i]
        if labelled:
            l = len(targets['labels'])
            max_targets = max(l, max_targets)
            for i in targets['labels']:
                labels.append(int(i))
            no_class += max(num_queries - l, 0)
            overshoot += min(num_queries - l, 0)
    d = dict(Counter(labels))
    return max_targets, d, no_class, overshoot

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    batch[1] = torch.stack(batch[1])
    # print(batch[0].size())
    return tuple(batch)


def load_data_paths(root_dir, file_pattern):
    paths = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in fnmatch.filter(filenames, file_pattern):
            paths.append(os.path.join(dirpath, filename))
    return paths



class NpzTensorDataset:
    """
    Loads all files into a list of dataframes and samples from them.
    """

    def __init__(self, root_dir=None, paths=None, file_pattern="*.npz", window_len=1000, p=0.1, num_classes = 3):
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.window_len = window_len
        self.participants_data_paths = load_data_paths(self.root_dir, self.file_pattern) if paths is None else paths
        self.nb_files = len(self.participants_data_paths)
        logging.info(f"Number of files: {self.nb_files}")
        self.dfs = [np.load(path) for path in self.participants_data_paths]
        #filter needed to avoid bug
        self.dfs = [df for df in self.dfs if len(df['EEG']) >= self.window_len]

        self.tensors = [torch.tensor(df['EEG'][:, :128], dtype=torch.float32) for df in
                        self.dfs]
        print(self.tensors[0].size())
        # self.num_classes = 3 #
        self.num_classes = num_classes
        if self.num_classes == 1:
            dict_map = {'L_fixation': 0, 'L_saccade': 1, 'L_blink': 2, 
                        'R_fixation': 0, 'R_saccade': 1, 'R_blink': 2, 'None': 3}
        elif self.num_classes == 2:
            dict_map = {'L_fixation': 0, 'L_saccade': 1, 'L_blink': 2,
                        'R_fixation': 0, 'R_saccade': 1, 'R_blink': 2, 'None': 3} #those above num_class will get mapped towards lower class labels
        elif self.num_classes == 3:
            dict_map = {'L_fixation': 0, 'L_saccade': 1, 'L_blink': 2,
                        'R_fixation': 0, 'R_saccade': 1, 'R_blink': 2, 'None': 3}
        else:
            raise Exception("Class Number not defined")
        unique_labels = [np.unique(df['labels']) for df in self.dfs]
        logging.info(np.concatenate(unique_labels).flatten())
        logging.info(np.unique(np.concatenate(unique_labels)).flatten())
        f = np.vectorize(dict_map.__getitem__)
        self.labels = [torch.tensor(f(df["labels"]).flatten(), dtype=torch.int) for df in
                       self.dfs]

        
        self.tensor_lengths = [len(tensor) for tensor in self.tensors]
        print(min(self.tensor_lengths))
        print(max(self.tensor_lengths))
        self.tensors = [self.cap(tensor) for tensor in self.tensors]
        self.max_vals = [tensor.max() for tensor in self.tensors]
        self.min_vals = [tensor.min() for tensor in self.tensors]
        self.max_offsets = [int(len(tensor) % self.window_len) for tensor in
                            self.tensors]  # max random offset for each tensor's start sample
        self.leftover = [len(tensor) % self.window_len for tensor in
                         self.tensors]  # leftover times after sampling window_len samples from tensor
        self.offsets = [random.randint(0, max_offset) for max_offset in
                        self.max_offsets]  # initialize random offsets for each tensor
        self.global_min = min(self.min_vals)
        self.global_max = max(self.max_vals)
        self.prefix_sum_samples = np.cumsum([length // self.window_len for length in
                                             self.tensor_lengths])  # prefix sum of each tensor's samples for fast sampling
        self.len = self.prefix_sum_samples[-1]

        self.labelled = torch.tensor(np.random.choice(2, self.len, p=[1 - p, p]))

        indices = torch.arange(0, self.len, 1)
        self.labelled_indices = indices[self.labelled == 1]
        self.unlabelled_indices = indices[self.labelled == 0]

        logging.info(f"Number of samples in dataset: {self.len}")
        assert self.len == self.prefix_sum_samples[
            -1], f"len {self.len} != prefix_sum_samples[-1] {self.prefix_sum_samples[-1]}"

        print(f"prefix sum samples {self.prefix_sum_samples}")
        
        # normalize and add constant channel to each tensor
        for i, tensor in enumerate(self.tensors):
            #tensor = self.cap(tensor) #cap tensor values
            tensor = min_max_normalize(tensor, self.min_vals[i], self.max_vals[i])
            val = (self.max_vals[i] - self.min_vals[i]) / (self.global_max - self.global_min)
            const = torch.ones((len(tensor), 1)) * val
            print(f"tensor shape {tensor.shape}")
            self.tensors[i] = torch.cat((tensor, const), dim=1)
            print(f"tensor shape {self.tensors[i].shape}")

        self.reset_offsets()
    
    def cap(self, tensor, threshold=150):
        tensor[tensor > threshold] = threshold
        tensor[tensor < -threshold] = threshold
        return tensor
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

        if self.offsets[file_idx] == self.leftover[file_idx]:
            tensor = self.tensors[file_idx][self.offsets[file_idx]:,:].view(-1, self.window_len, 129)[sample_idx, :, :]
        else:
            tensor = self.tensors[file_idx][self.offsets[file_idx]:-(self.leftover[file_idx] - self.offsets[file_idx]),
                 :].view(-1, self.window_len, 129)[sample_idx, :, :]

        if torch.isnan(tensor).any():
            logging.info(f"NaN in tensor from {self.participants_data_paths[file_idx]}")
            raise ValueError

        label = {'boxes': torch.zeros((1, 2)), 'labels': torch.Tensor([self.num_classes])}
        if self.labelled[idx]:
            if self.offsets[file_idx] == self.leftover[file_idx]:
                label = self.labels[file_idx][self.offsets[file_idx]:].view(-1, self.window_len)[sample_idx, :]
            else:
                label = self.labels[file_idx][
                    self.offsets[file_idx]:-(self.leftover[file_idx] - self.offsets[file_idx])].view(-1,
                                                                                                     self.window_len)[
                    sample_idx, :]

            label = create_annotations_num_classes(label, self.num_classes, 3)

        # check if nan in tensor

        return self.labelled[idx], tensor, label


class TwoSubsetSamplers(Sampler):

    def __init__(self, batch_size: int, indices1: Sequence[int], indices2: Sequence[int], generator=None) -> None:
        self.indices1 = indices1  # labelled
        self.indices2 = indices2  # unlabelled
        self.batch_size = batch_size
        self.generator = generator

        print(batch_size)
        self.offset1 = len(self.indices1) // batch_size * batch_size
        print(f"Available labelled samples {len(self.indices1)}")
        # self.offset2 = len(self.indices2) // batch_size * batch_size
        self.offset2 = batch_size
        print(f"Available unlabelled samples {len(self.indices2)}")
        self.max_offset2 = len(self.indices2) // batch_size * batch_size
        print(f"Actually available unlabelled samples {self.max_offset2}")

    def __iter__(self):
        temp1 = self.indices1[torch.randperm(len(self.indices1))][:self.offset1]
        print(f"Current labelled batches: {temp1.size(0) // self.batch_size}")
        temp2 = self.indices2[torch.randperm(len(self.indices2))][:self.offset2]
        print(f"Current unlabelled batches: {temp2.size(0) // self.batch_size}")
        indices = torch.cat([temp1, temp2])
        # print(indices.size())
        self.indices = indices.view(-1, self.batch_size)[torch.randperm(len(indices) // self.batch_size)]
        # print(self.indices.size())
        print(f"Current total batches {self.indices.size(0)}")

        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return (self.offset1 + self.offset2) // self.batch_size

    def reset(self):
        self.offset2 = self.batch_size

    def set_zero(self):
        logging.info(f"Disabling unlabelled batches ")
        self.offset2 = 0

    def set_one(self):
        logging.info(f"Enabling unlabelled batches ")

        self.offset2 = self.batch_size

    def set_max(self):
        self.offset2 = self.max_offset2

    def next_epoch(self):
        logging.info(f"Doubling unlabelled batches ")

        self.offset2 = min(self.offset2 * 2, self.max_offset2)
        print(f"Setting unlabelled batches to {self.offset2}")


class NpzMultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, file_pattern="*.npz", window_len=1000, batch_size=32, p=0.5, num_classes = 3):
        super().__init__()

        logging.info("Requires Precut Samples")
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.window_len = window_len
        self.batch_size = batch_size
        self.p = p

        self.num_classes = num_classes
        # TODO: make this more general
        dirs = [os.path.join(root_dir, o) for o in os.listdir(root_dir)
                if os.path.isdir(os.path.join(root_dir, o))]
        dirs = [os.path.join(o, p) for o in dirs for p in os.listdir(o)
                 if os.path.isdir(os.path.join(o, p))]
        print(dirs)
        dirs = [o for o in dirs if o != root_dir]
        print(dirs)
        print(os.listdir(root_dir))
        print(os.listdir("/itet-stor/ljie/deepeye_itetnas04/semester-project-djl/datasets/ICML_debug"))
        print(dirs)
        print(root_dir)
        print(file_pattern)
        logging.info(dirs)

        nb_dirs = len(dirs)
        print(nb_dirs)
        np.random.seed(42)

        # np.random.seed(109)
        nums = np.random.permutation(nb_dirs)
        train_dirs = [dirs[i] for i in nums[:int(0.8 * nb_dirs)]]
        val_dirs = [dirs[i] for i in nums[int(0.8 * nb_dirs):int(0.9 * nb_dirs)]]
        test_dirs = [dirs[i] for i in nums[int(0.9 * nb_dirs):]]

        print(train_dirs)
        print(val_dirs)
        print(test_dirs)

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

        print(f"Found {len(self.train_paths)} training files.")
        print(f"Found {len(self.val_paths)} validation files.")
        print(f"Found {len(self.test_paths)} test files.")

    def prepare_data(self):
        pass

    def setup(self, stage: str = None):
        logging.info(f"Setting up datamodule with stage {stage}")

        if stage == "fit" or stage is None:
            self.train = NpzTensorDataset(paths=self.train_paths, file_pattern=self.file_pattern,
                                              window_len=self.window_len, p=self.p, num_classes=self.num_classes)
            # self.train = VirtualTensorDataset(root_dir=self.root_dir, file_pattern=self.file_pattern,
            #                                   window_len=self.window_len)

            self.train_sampler = TwoSubsetSamplers(self.batch_size, indices1=self.train.labelled_indices,
                                                   indices2=self.train.unlabelled_indices)
            logging.info(f"Datamodule: Created training dataset with {len(self.train)} samples")
            self.val = NpzTensorDataset(paths=self.val_paths, file_pattern=self.file_pattern,
                                            window_len=self.window_len, p=1.0, num_classes=self.num_classes)
            # self.val = VirtualTensorDataset(root_dir=self.root_dir, file_pattern=self.file_pattern,
            #                                   window_len=self.window_len)

            self.val_sampler = TwoSubsetSamplers(self.batch_size, indices1=self.val.labelled_indices,
                                                 indices2=self.val.unlabelled_indices)
            self.val_sampler.set_max()
            logging.info(f"Datamodule: Created validation dataset with {len(self.val)} samples")

        if stage == "test" or stage is None:
            self.test = NpzTensorDataset(paths=self.test_paths, file_pattern=self.file_pattern,
                                             window_len=self.window_len, p=1.0, num_classes=self.num_classes)
            # self.test = VirtualTensorDataset(root_dir=self.root_dir, file_pattern=self.file_pattern,
            #                                   window_len=self.window_len)

            self.test_sampler = TwoSubsetSamplers(self.batch_size, indices1=self.test.labelled_indices,
                                                  indices2=self.test.unlabelled_indices)
            self.test_sampler.set_max()
            logging.info(f"Datamodule: Created test dataset with {len(self.test)} samples")

        logging.info(f"Datamodule setup stage {stage} done.")


    def get_train_class_weights(self, num_queries):

        max_targets, d, no_class, overshoot = dataset_summaries(self.train, num_queries)
        d[self.train.num_classes] = no_class
        logging.info("Train Set")
        logging.info(f"Max targets {max_targets}")
        logging.info(f"Label Distribution {d}")
        logging.info(f"Overshoot {overshoot}")

        max_targets, d, no_class, overshoot = dataset_summaries(self.val, num_queries)
        d[self.train.num_classes] = no_class
        logging.info("Validation Set")
        logging.info(f"Max targets {max_targets}")
        logging.info(f"Label Distribution {d}")
        logging.info(f"Overshoot {overshoot}")

        max_targets, d, no_class, overshoot = dataset_summaries(self.test, num_queries)
        d[self.train.num_classes] = no_class
        logging.info("Validation Set")
        logging.info(f"Max targets {max_targets}")
        logging.info(f"Label Distribution {d}")
        logging.info(f"Overshoot {overshoot}")

    def train_dataloader(self):
        # return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, sampler=self.train_sampler, collate_fn=collate_fn)
        return DataLoader(self.train, num_workers=0, batch_sampler=self.train_sampler, collate_fn=collate_fn)

    def val_dataloader(self):
        # return DataLoader(self.val, batch_size=self.batch_size, num_workers=4, sampler=self.val_sampler, collate_fn=collate_fn)
        return DataLoader(self.val, num_workers=0, batch_sampler=self.val_sampler, collate_fn=collate_fn)

    def test_dataloader(self):
        # return DataLoader(self.test, batch_size=self.batch_size, num_workers=4, sampler=self.test_sampler, collate_fn=collate_fn)
        return DataLoader(self.test, num_workers=0, batch_sampler=self.test_sampler, collate_fn=collate_fn)
