import torch
from torch.utils.data import DataLoader, Sampler
import pytorch_lightning as pl
import random
import pandas as pd
from torch.utils.data import WeightedRandomSampler
import sklearn.preprocessing as preprocessing
import os, fnmatch
import logging
from util.box_ops import box_cxw_to_xlxh, box_xlxh_to_cxw
import numpy as np
from collections import Counter

# from dataset.dataset import create_annotations_num_classes
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


class VirtualTensorDataset:
    """
    Loads all files into a list of dataframes and samples from them.
    """

    def __init__(self, root_dir=None, paths=None, file_pattern="*stream*.csv", window_len=1000, p=0.1, num_classes=3):
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.window_len = window_len
        self.participants_data_paths = load_data_paths(self.root_dir, self.file_pattern) if paths is None else paths
        self.nb_files = len(self.participants_data_paths)
        logging.info(f"Number of files: {self.nb_files}")
        self.dfs = [pd.read_csv(path) for path in self.participants_data_paths]
        # filter needed to avoid bug
        self.dfs = [df for df in self.dfs if len(df.index) >= self.window_len]

        self.tensors = [torch.tensor(df.loc[:, "channel_0":"channel_127"].values, dtype=torch.float32) for df in
                        self.dfs]

        # self.num_classes = 3 #
        self.num_classes = num_classes
        if self.num_classes == 1:
            dict_map = {'L_fi': 0, 'L_sa': 1, 'L_bl': 2,
                        'R_fi': 0, 'R_sa': 1, 'R_bl': 2, 'None': 3}
        elif self.num_classes == 2:
            dict_map = {'L_fi': 0, 'L_sa': 1, 'L_bl': 2,
                        'R_fi': 0, 'R_sa': 1, 'R_bl': 2,
                        'None': 3}  # those above num_class will get mapped towards lower class labels
        elif self.num_classes == 3:
            dict_map = {'L_fi': 0, 'L_sa': 1, 'L_bl': 2,
                        'R_fi': 0, 'R_sa': 1, 'R_bl': 2, 'None': 3}
        else:
            raise Exception("Class Numer not defined")
        unique_labels = [df.loc[:, "event"].unique() for df in self.dfs]
        logging.info(np.concatenate(unique_labels).flatten())
        logging.info(np.unique(np.concatenate(unique_labels)).flatten())
        self.labels = [torch.tensor(df.loc[:, "event"].map(dict_map).values, dtype=torch.int) for df in
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
            # cap tensor values
            tensor = min_max_normalize(tensor)
            val = (self.max_vals[i] - self.min_vals[i]) / (self.global_max - self.global_min)
            const = torch.ones((len(tensor), 1)) * val
            # print(f"tensor shape {tensor.shape}")
            self.tensors[i] = torch.cat((tensor, const), dim=1)
            # print(f"tensor shape {self.tensors[i].shape}")

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
            tensor = self.tensors[file_idx][self.offsets[file_idx]:, :].view(-1, self.window_len, 129)[sample_idx, :, :]
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


class MultiTaskDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, file_pattern="*stream*.csv", window_len=1000, batch_size=32, p=0.5, num_classes=3):
        super().__init__()
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
        logging.info(dirs)

        nb_dirs = len(dirs)
        np.random.seed(42)

        # np.random.seed(109)
        nums = np.random.permutation(nb_dirs)
        train_dirs = [dirs[i] for i in nums[:int(0.8 * nb_dirs)]]
        val_dirs = [dirs[i] for i in nums[int(0.8 * nb_dirs):int(0.9 * nb_dirs)]]
        test_dirs = [dirs[i] for i in nums[int(0.9 * nb_dirs):]]

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

    def setup(self, stage: str = None):
        logging.info(f"Setting up datamodule with stage {stage}")

        if stage == "fit" or stage is None:
            self.train = VirtualTensorDataset(paths=self.train_paths, file_pattern=self.file_pattern,
                                              window_len=self.window_len, p=self.p, num_classes=self.num_classes)
            # self.train = VirtualTensorDataset(root_dir=self.root_dir, file_pattern=self.file_pattern,
            #                                   window_len=self.window_len)

            self.train_sampler = TwoSubsetSamplers(self.batch_size, indices1=self.train.labelled_indices,
                                                   indices2=self.train.unlabelled_indices)
            logging.info(f"Datamodule: Created training dataset with {len(self.train)} samples")
            self.val = VirtualTensorDataset(paths=self.val_paths, file_pattern=self.file_pattern,
                                            window_len=self.window_len, p=1.0, num_classes=self.num_classes)
            # self.val = VirtualTensorDataset(root_dir=self.root_dir, file_pattern=self.file_pattern,
            #                                   window_len=self.window_len)

            self.val_sampler = TwoSubsetSamplers(self.batch_size, indices1=self.val.labelled_indices,
                                                 indices2=self.val.unlabelled_indices)
            self.val_sampler.set_max()
            logging.info(f"Datamodule: Created validation dataset with {len(self.val)} samples")

        if stage == "test" or stage is None:
            self.test = VirtualTensorDataset(paths=self.test_paths, file_pattern=self.file_pattern,
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


def unbiassed_weights(tensor_y, num_boxes=None, max_queries=10):
    """
    :param tensor_y: (samples, seq length, ) np array
    :return: weights: DoubleTensor with sampling weights
    """
    # seq_length = len(tensor_y[0])
    # logging.info("Output seq_length: {}".format(seq_length))
    num_samples = len(tensor_y)
    arr = np.ones(num_samples)

    if num_boxes is not None:
        logging.info("Setting samples > max_queries to 0")
        arr[num_boxes > max_queries] = 0
        logging.info(f"Number of elements over queries {np.sum(num_boxes > max_queries)}")
    return torch.DoubleTensor(arr.tolist())


def biased_weights(tensor_y, validation=False, num_boxes=None, max_queries=10):
    """
    creates weight tensor biased towards minority classes
    :param tensor_y: (samples, seq length, ) np array
    :return: weights: DoubleTensor with sampling weights
    """
    # seq_length = len(tensor_y[0])
    num_samples = len(tensor_y)
    arr = np.ones(num_samples)

    if not validation:
        """
        follow the simple strategy of assigning higher weights to samples
        containing a certain amount of minority samples
        """
        f1 = lambda x: np.count_nonzero(x == 1) > 40
        # mask = np.apply_along_axis(f1, -1, tensor_y)
        mask = np.array([f1(x) for x in tensor_y])
        logging.info(f"# of time steps with saccades > 40: {np.count_nonzero(mask)}")
        arr[mask] = 2

        f2 = lambda x: np.count_nonzero(x == 2) > 20
        # mask1 = np.apply_along_axis(f2,-1, tensor_y)
        mask1 = np.array([f2(x) for x in tensor_y])
        logging.info(f"# of samples time steps > 20 blinks: {np.count_nonzero(mask1)}")
        arr[mask1] = 4

    if num_boxes is not None:
        arr[num_boxes > max_queries] = 0
        n_lost = np.sum(num_boxes > max_queries)
        logging.info(f"# of elements with more regions than available queries {n_lost}")

    return torch.DoubleTensor(arr.tolist())


def get_num_boxes(y, num_classes):
    """
    iterates over y and creates box annotations
    :param y: (seq length, ) np array
    :return: dict containing target values
    """
    i = 0
    end = len(y)
    num_boxes = 0
    class_counts = np.zeros(num_classes)
    while i < end:
        label = y[i]

        while y[i] == label and i < end - 1:  # extend event
            i += 1
        """
        for 2 classes:
        if label != 1:  # skip saccades
            num_boxes += 1
        """
        class_counts[label] += 1
        num_boxes += 1

        i += 1

    return num_boxes, class_counts


def collate_boxes(y, num_classes):
    # num_boxes = np.apply_along_axis(get_num_boxes, -1, y)
    num_boxes = []
    class_counts = np.zeros(num_classes)
    for y_i in y:
        num_boxes_i, class_counts_i = get_num_boxes(y_i, num_classes)
        class_counts += class_counts_i
        num_boxes.append(num_boxes_i)
    num_boxes = np.array(num_boxes)
    return num_boxes, class_counts


def create_annotations(y):
    """
    iterates over y and creates box annotations
    :param y: (seq length, ) np array
    :return: dict containing target values
    """
    i = 0
    end = len(y)
    bboxes = []
    labels = []
    regions =  torch.zeros(len(y))
    region_idx = 0
    while i < end:
        left = i
        label = y[i]

        while i < end and y[i] == label:  # extend event
            i += 1
        right = i

        regions[left:right] = region_idx
        region_idx += 1
        # create a box of the form [x_low, x_high] in normalized form

        """
        for 2 classes: 
        if label != 1: # skip saccades
            box = torch.Tensor([left / float(end), right / float(end)])
            bboxes.append(box)
            labels.append(torch.tensor(0.) if label == 0 else torch.tensor(1))
        """
        box = torch.Tensor([left / float(end), right / float(end)])
        if label >= 0: #guarantee legal label
            bboxes.append(box)
            labels.append(label.clone().detach())  # if label == 0 else torch.tensor(1))

    if len(bboxes) > 0:
        # stack boxes in a tensor and convert boxes to [center, width]
        boxes = torch.stack(bboxes)
        boxes = box_xlxh_to_cxw(boxes)
    else:
        boxes = torch.Tensor()

    return {"boxes": boxes, "labels": torch.as_tensor(labels)}, regions


def create_annotations_num_classes(y, num_classes, max_classes, positions=None):
    """
    iterates over y and creates box annotations
    :param y: (seq length, ) np array
    :return: dict containing target values
    """
    i = 0
    end = len(y)
    bboxes = []
    labels = []
    position_labels = [] if positions is not None else None
    relative_labels = [] if positions is not None else None
    while i < end:

        left = i
        label = y[i]

        while i < end and y[i] == label:  # extend event
            i += 1
        right = i

        # create a box of the form [x_low, x_high] in normalized form

        """
        for 2 classes: 
        if label != 1: # skip saccades
            box = torch.Tensor([left / float(end), right / float(end)])
            bboxes.append(box)
            labels.append(torch.tensor(0.) if label == 0 else torch.tensor(1))
        """
        box = torch.Tensor([left / float(end), right / float(end)])
        if label < num_classes:
            bboxes.append(box)
            labels.append(torch.tensor(label))  # if label == 0 else torch.tensor(1))
        elif label >= num_classes and label < max_classes:
            bboxes.append(box)
            labels.append(torch.tensor(num_classes - 1))

        if position_labels is not None:
            position_labels.append(torch.Tensor(np.mean(positions[left: right], axis=0)))

        if relative_labels is not None:
            relative_labels.append(torch.tensor(positions[right - 1] - positions[left], dtype=torch.float))

    if len(bboxes) > 0:
        # stack boxes in a tensor and convert boxes to [center, width]
        boxes = torch.stack(bboxes)
        boxes = box_xlxh_to_cxw(boxes)
    else:
        boxes = torch.Tensor()

    if position_labels is None:
        position_labels = torch.Tensor()
    else:
        position_labels = torch.stack(position_labels)

    if relative_labels is None:
        relative_labels = torch.Tensor()
    else:
        relative_labels = torch.stack(relative_labels)
    return {'boxes': boxes, 'labels': torch.as_tensor(labels), 'positions': torch.as_tensor(position_labels),
            'relative_change': torch.as_tensor(relative_labels)}


def create_annotations_num_classes(y, num_classes, max_classes):
    """
    iterates over y and creates box annotations
    :param y: (seq length, ) np array
    :return: dict containing target values
    """
    i = 0
    end = len(y)
    bboxes = []
    labels = []
    position_labels = [] if positions is not None else None
    relative_labels = [] if positions is not None else None
    while i < end:
        left = i
        label = y[i]

        while i < end and y[i] == label:  # extend event
            i += 1
        right = i

        # create a box of the form [x_low, x_high] in normalized form

        """
        for 2 classes: 
        if label != 1: # skip saccades
            box = torch.Tensor([left / float(end), right / float(end)])
            bboxes.append(box)
            labels.append(torch.tensor(0.) if label == 0 else torch.tensor(1))
        """
        box = torch.Tensor([left / float(end), right / float(end)])
        if label < num_classes:
            bboxes.append(box)
            labels.append(torch.tensor(label))  # if label == 0 else torch.tensor(1))
        elif label >= num_classes and label < max_classes:
            bboxes.append(box)
            labels.append(torch.tensor(num_classes - 1))

        if position_labels is not None:
            position_labels.append(torch.Tensor(np.mean(positions[left: right], axis=0)))

        if relative_labels is not None:
            relative_labels.append(torch.tensor(positions[right - 1] - positions[left], dtype=torch.float))

    if len(bboxes) > 0:
        # stack boxes in a tensor and convert boxes to [center, width]
        boxes = torch.stack(bboxes)
        boxes = box_xlxh_to_cxw(boxes)
    else:
        boxes = torch.Tensor()

    if position_labels is None:
        position_labels = torch.Tensor()
    else:
        position_labels = torch.stack(position_labels)

    if relative_labels is None:
        relative_labels = torch.Tensor()
    else:
        relative_labels = torch.stack(relative_labels)
    return {'boxes': boxes, 'labels': torch.as_tensor(labels), 'positions': torch.as_tensor(position_labels),
            'relative_change': torch.as_tensor(relative_labels)}


# def collate_fn(y):
#    labels = np.apply_along_axis(create_annotations, -1, y)
#    return labels


class TensorListDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, positions=None):
        """
        :param X: iterable implementing __get__
        :param y: iterable implementing __get__
        """
        super(TensorListDataset, self).__init__()
        self.X = X
        self.y = y
        self.positions = positions
        self.length = X.size(0)
        self.labelled = torch.ones((self.length,))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        X_idx = self.X[idx]
        if self.positions is None:
            annotations = create_annotations_num_classes(self.y[idx], 3, 3)  # to help ignore possible illegal classes
        else:
            annotations = create_annotations_num_classes(self.y[idx], 3, 3, positions=self.positions[idx])
        # boxes = annotations['boxes']
        # labels = annotations['labels']
        return self.labelled[idx], X_idx, annotations  # {'X':X_idx, 'boxes':boxes, 'labels':labels}


class WeightedRandomSamplerWrapper(Sampler):

    def __init__(self, sampler) -> None:
        self.sampler = sampler

    def __iter__(self):
        return sampler.__iter__()

    def __len__(self):
        return sampler.__len__()

    def reset(self):
        pass

    def set_zero(self):
        pass

    def set_one(self):
        pass

    def set_max(self):
        pass

    def next_epoch(self):
        pass


def create_dataloader_cut129(
        data_dir,
        file,
        validation=False,
        batch_size=32,
        workers=1,
        collate_fn=None,
        standard_scale=False,
        scaler=None,
        max_queries=10,
        num_classes=3,
        num_timesteps=500,
        apply_label_dict=False,
):
    """
    creates the dataloader for the DETRtime pipeline
    :param collate_fn: merges a list of samples to form a batch
    :param apply_label_dict:
    :param sleep: boolean
    :param num_classes: int
    :param max_queries: int
    :param scaler: implementing sklearn.Transformer API
    :param validation: bool
    :param data_dir: str
    :param file: str
    :param batch_size: int
    :param workers: int
    :return:
    """
    logging.info("Loading data")
    df = pd.read_csv(f"{data_dir}/{file}")
    print(df.columns)
    logging.info("Finished loading file")
    # load and process data
    logging.info("Preprocessing data")
    X = df.loc[:, "channel_0":"channel_128"].values
    logging.info(X.shape)
    X = X.reshape(-1, num_timesteps, 129)

    dict_map = {
        "L_fi": 0,
        "L_sa": 1,
        "L_bl": 2,
        "R_fi": 0,
        "R_sa": 1,
        "R_bl": 2,
        "None": 3
    }

    y = df.loc[:, "event"].map(dict_map).values
    y = y.reshape(-1, num_timesteps)

    positions = df.loc[:, "x":"y"].values
    positions = positions / 800  # scaling
    positions = positions.reshape(-1, num_timesteps, 2)

    X_max = np.amax(np.abs(X), axis=2)
    X_max = np.amax(X_max, axis=1)
    mask = X_max <= 150

    X = X[mask]

    y = y[mask]

    positions = positions[mask]

    logging.info(f"Removed Samples: X:{X.shape}, y:{y.shape}")

    if standard_scale:
        logging.info("Rescaling data")
        if not validation and scaler is not None:
            ls, seq_length, channel = X.shape
            X = X.reshape(-1, channel)
            X = scaler.fit_transform(X)
            X = X.reshape(-1, seq_length, channel)
        elif scaler is not None:
            ls, seq_length, channel = X.shape
            X = X.reshape(-1, channel)
            X = scaler.transform(X)
            X = X.reshape(-1, seq_length, channel)

    logging.info("Annotating labels")

    logging.info("Using normal annotations")

    y = y.astype("int")
    logging.info(f'Unique labels {np.unique(y)}')
    y = torch.as_tensor(y)
    tensor_x = torch.as_tensor(X).float()

    dataset = TensorListDataset(tensor_x, y, positions=positions)

    if not validation:
        logging.info("Creating training dataloader")
        num_boxes = None
        weights = biased_weights(y, num_boxes=num_boxes, max_queries=max_queries)
        sampler = WeightedRandomSampler(weights, len(weights))
        return scaler, torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
    else:
        logging.info("Creating validation dataloader")
        # Generally no biased sampling of any sort in validation
        # if sleep:
        #     weights = unbiassed_weights(y, validation=True, num_boxes=num_boxes, max_queries=max_queries)
        # else:
        #     weights = biased_weights(y, validation=True, num_boxes=num_boxes, max_queries=max_queries)
        # sampler = WeightedRandomSampler(weights, len(weights))
        return scaler, torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=collate_fn,
            drop_last=True,
        )


def create_dataloader_cut(
        data_dir,
        file,
        validation=False,
        batch_size=32,
        workers=1,
        collate_fn=None,
        standard_scale=False,
        scaler=None,
        max_queries=10,
        num_classes=3,
        num_timesteps=500,
        apply_label_dict=False,
):
    """
    creates the dataloader for the DETRtime pipeline
    :param collate_fn: merges a list of samples to form a batch
    :param apply_label_dict:
    :param sleep: boolean
    :param num_classes: int
    :param max_queries: int
    :param scaler: implementing sklearn.Transformer API
    :param validation: bool
    :param data_dir: str
    :param file: str
    :param batch_size: int
    :param workers: int
    :return:
    """
    logging.info("Loading data")
    df = pd.read_csv(f"{data_dir}/{file}")
    print(df.columns)
    logging.info("Finished loading file")
    # load and process data
    logging.info("Preprocessing data")
    X = df.loc[:, "channel_0":"channel_127"].values
    logging.info(X.shape)

    x_mean = np.expand_dims(np.mean(X, axis=1), axis=1)
    X = np.concatenate([X, x_mean], axis=1)
    logging.info(f"Addded mean channel {X.shape}")
    X = X.reshape(-1, num_timesteps, 129)

    dict_map = {
        "L_fi": 0,
        "L_sa": 1,
        "L_bl": 2,
        "R_fi": 0,
        "R_sa": 1,
        "R_bl": 2,
        "None": 3
    }

    y = df.loc[:, "event"].map(dict_map).values
    y = y.reshape(-1, num_timesteps)

    positions = df.loc[:, "x":"y"].values
    positions = positions / 800  # scaling
    positions = positions.reshape(-1, num_timesteps, 2)

    X_max = np.amax(np.abs(X), axis=2)
    X_max = np.amax(X_max, axis=1)
    mask = X_max <= 150

    X = X[mask]

    y = y[mask]

    positions = positions[mask]

    logging.info(f"Removed Samples: X:{X.shape}, y:{y.shape}")

    if standard_scale:
        logging.info("Rescaling data")
        if not validation and scaler is not None:
            ls, seq_length, channel = X.shape
            X = X.reshape(-1, channel)
            X = scaler.fit_transform(X)
            X = X.reshape(-1, seq_length, channel)
        elif scaler is not None:
            ls, seq_length, channel = X.shape
            X = X.reshape(-1, channel)
            X = scaler.transform(X)
            X = X.reshape(-1, seq_length, channel)

    logging.info("Annotating labels")

    logging.info("Using normal annotations")

    y = y.astype("int")
    logging.info(f'Unique labels {np.unique(y)}')
    y = torch.as_tensor(y)
    tensor_x = torch.as_tensor(X).float()

    dataset = TensorListDataset(tensor_x, y, positions=positions)

    if not validation:
        logging.info("Creating training dataloader")
        num_boxes = None
        weights = biased_weights(y, num_boxes=num_boxes, max_queries=max_queries)
        sampler = WeightedRandomSampler(weights, len(weights))
        return scaler, torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
    else:
        logging.info("Creating validation dataloader")
        # Generally no biased sampling of any sort in validation
        # if sleep:
        #     weights = unbiassed_weights(y, validation=True, num_boxes=num_boxes, max_queries=max_queries)
        # else:
        #     weights = biased_weights(y, validation=True, num_boxes=num_boxes, max_queries=max_queries)
        # sampler = WeightedRandomSampler(weights, len(weights))
        return scaler, torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=collate_fn,
            drop_last=True,
        )


def create_dataloader(
        data_dir,
        file,
        validation=False,
        batch_size=32,
        workers=1,
        collate_fn=None,
        standard_scale=False,
        scaler=None,
        max_queries=10,
        num_classes=3,
        apply_label_dict=False,
):
    """
    creates the dataloader for the DETRtime pipeline
    currently handles SleepEDF and EEG eye movement segmentation datasets
    :param collate_fn: merges a list of samples to form a batch
    :param apply_label_dict:
    :param sleep: boolean
    :param num_classes: int
    :param max_queries: int
    :param scaler: implementing sklearn.Transformer API
    :param validation: bool
    :param data_dir: str
    :param file: str
    :param batch_size: int
    :param workers: int
    :return:
    """
    logging.info("Loading data")
    data = np.load(f"{data_dir}/{file}")
    logging.info("Finished loading file")
    # load and process data
    logging.info("Preprocessing data")
    X = data["EEG"].astype(float)
    if standard_scale:
        logging.info("Rescaling data")
        if not validation and scaler is not None:
            ls, seq_length, channel = X.shape
            X = X.reshape(-1, channel)
            X = scaler.fit_transform(X)
            X = X.reshape(-1, seq_length, channel)
        elif scaler is not None:
            ls, seq_length, channel = X.shape
            X = X.reshape(-1, channel)
            X = scaler.transform(X)
            X = X.reshape(-1, seq_length, channel)

    logging.info("Annotating labels")

    logging.info("Using normal annotations")
    conv = {
        "L_fixation": 0,
        "L_saccade": 1,
        "L_blink": 2,
        "R_fixation": 0,
        "R_saccade": 1,
        "R_blink": 2,
    }
    func = np.vectorize(conv.get, otypes=["int"])
    if apply_label_dict:
        logging.info("Applying labeling dict")
        y = func(data["labels"])
    else:
        logging.info("Not applying labeling dict")
        y = data["labels"]

    y = y.astype("int")
    # logging.info(f'Unique labels {np.unique(y)}')
    y = torch.as_tensor(y)
    tensor_x = torch.as_tensor(X).float()

    dataset = TensorListDataset(tensor_x, y)

    if not validation:
        logging.info("Creating training dataloader")
        # Calculating number of regions within each sample
        # num_boxes, class_counts = collate_boxes(y, num_classes)  # dict collation
        # for i, c in enumerate(class_counts):
        #     logging.info(f'Class {i}: #{c} boxes')
        # # of regions preprocessed
        # logging.info("Getting num_boxes")
        # num_boxes = data['num_boxes']
        # # of boxes not used
        num_boxes = None

        weights = biased_weights(y, num_boxes=num_boxes, max_queries=max_queries)
        sampler = WeightedRandomSampler(weights, len(weights))
        return scaler, torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
    else:
        logging.info("Creating validation dataloader")
        # Generally no biased sampling of any sort in validation
        # if sleep:
        #     weights = unbiassed_weights(y, validation=True, num_boxes=num_boxes, max_queries=max_queries)
        # else:
        #     weights = biased_weights(y, validation=True, num_boxes=num_boxes, max_queries=max_queries)
        # sampler = WeightedRandomSampler(weights, len(weights))
        return scaler, torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=workers,
            collate_fn=collate_fn,
            drop_last=True,
        )


class TensorDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=32, p=0.5, scaler="standard", num_classes=3, max_queries=10):
        super().__init__()

        self.data_dir = data_dir

        self.batch_size = batch_size
        self.p = p
        self.max_queries = max_queries
        self.num_classes = num_classes

        if scaler == "standard":
            self.standard_scale = True
            self.scaler = preprocessing.StandardScaler()
        elif scaler == "minmax":
            self.standard_scale = True
            self.scaler = preprocessing.MinMaxScaler()
        elif scaler == "none":
            self.standard_scale = False
            self.scaler = None
        else:
            raise Exception(f"Scaler {scaler} not implemented")

    def prepare_data(self):
        pass

    def setup(self, stage: str = None, cutting_mode: str = "precut"):
        logging.info(f"Setting up datamodule with stage {stage}")
        logging.info("Debug test set")
        if cutting_mode == "precut":
            if stage == "fit" or stage is None:
                _, self.train = create_dataloader(data_dir=self.data_dir, file="train.npz", validation=False,
                                                  batch_size=self.batch_size, workers=2, collate_fn=collate_fn,
                                                  standard_scale=self.standard_scale, scaler=self.scaler,
                                                  max_queries=self.max_queries,
                                                  num_classes=self.num_classes, apply_label_dict=True)
                self.train_sampler = WeightedRandomSamplerWrapper(self.train.sampler)
                logging.info(f"Datamodule: Created training dataset with {len(self.train)} samples")
                _, self.val = create_dataloader(data_dir=self.data_dir, file="val.npz", validation=False,
                                                batch_size=self.batch_size, workers=1, collate_fn=collate_fn,
                                                standard_scale=self.standard_scale, scaler=self.scaler,
                                                max_queries=self.max_queries,
                                                num_classes=self.num_classes, apply_label_dict=True)
                self.val_sampler = WeightedRandomSamplerWrapper(self.val.sampler)
                logging.info(f"Datamodule: Created validation dataset with {len(self.val)} samples")

            if stage == "test" or stage is None:
                _, self.test = create_dataloader(data_dir=self.data_dir, file="test.npz", validation=False,
                                                 batch_size=self.batch_size, workers=1, collate_fn=collate_fn,
                                                 standard_scale=self.standard_scale, scaler=self.scaler,
                                                 max_queries=self.max_queries,
                                                 num_classes=self.num_classes, apply_label_dict=True)
                self.test_sampler = WeightedRandomSamplerWrapper(self.test.sampler)

                logging.info(f"Datamodule: Created test dataset with {len(self.test)} samples")
        elif cutting_mode == "uncut":
            if stage == "fit" or stage is None:
                _, self.train = create_dataloader_cut(data_dir=self.data_dir, file="train.csv", validation=False,
                                                      batch_size=self.batch_size, workers=2, collate_fn=collate_fn,
                                                      standard_scale=self.standard_scale, scaler=self.scaler,
                                                      max_queries=self.max_queries,
                                                      num_classes=self.num_classes, apply_label_dict=True)
                self.train_sampler = WeightedRandomSamplerWrapper(self.train.sampler)
                logging.info(f"Datamodule: Created training dataset with {len(self.train)} samples")
                _, self.val = create_dataloader_cut(data_dir=self.data_dir, file="val.csv", validation=False,
                                                    batch_size=self.batch_size, workers=1, collate_fn=collate_fn,
                                                    standard_scale=self.standard_scale, scaler=self.scaler,
                                                    max_queries=self.max_queries,
                                                    num_classes=self.num_classes, apply_label_dict=True)
                self.val_sampler = WeightedRandomSamplerWrapper(self.val.sampler)
                logging.info(f"Datamodule: Created validation dataset with {len(self.val)} samples")

            if stage == "test" or stage is None:
                _, self.test = create_dataloader_cut(data_dir=self.data_dir, file="test.csv", validation=False,
                                                     batch_size=self.batch_size, workers=1, collate_fn=collate_fn,
                                                     standard_scale=self.standard_scale, scaler=self.scaler,
                                                     max_queries=self.max_queries,
                                                     num_classes=self.num_classes, apply_label_dict=True)
                self.test_sampler = WeightedRandomSamplerWrapper(self.test.sampler)

                logging.info(f"Datamodule: Created test dataset with {len(self.test)} samples")
        elif cutting_mode == "uncut129":
            if stage == "fit" or stage is None:
                _, self.train = create_dataloader_cut129(data_dir=self.data_dir, file="train.csv", validation=False,
                                                         batch_size=self.batch_size, workers=2, collate_fn=collate_fn,
                                                         standard_scale=self.standard_scale, scaler=self.scaler,
                                                         max_queries=self.max_queries,
                                                         num_classes=self.num_classes, apply_label_dict=True)
                self.train_sampler = WeightedRandomSamplerWrapper(self.train.sampler)
                logging.info(f"Datamodule: Created training dataset with {len(self.train)} samples")
                _, self.val = create_dataloader_cut129(data_dir=self.data_dir, file="val.csv", validation=False,
                                                       batch_size=self.batch_size, workers=1, collate_fn=collate_fn,
                                                       standard_scale=self.standard_scale, scaler=self.scaler,
                                                       max_queries=self.max_queries,
                                                       num_classes=self.num_classes, apply_label_dict=True)
                self.val_sampler = WeightedRandomSamplerWrapper(self.val.sampler)
                logging.info(f"Datamodule: Created validation dataset with {len(self.val)} samples")

            if stage == "test" or stage is None:
                _, self.test = create_dataloader_cut129(data_dir=self.data_dir, file="test.csv", validation=False,
                                                        batch_size=self.batch_size, workers=1, collate_fn=collate_fn,
                                                        standard_scale=self.standard_scale, scaler=self.scaler,
                                                        max_queries=self.max_queries,
                                                        num_classes=self.num_classes, apply_label_dict=True)
                self.test_sampler = WeightedRandomSamplerWrapper(self.test.sampler)

                logging.info(f"Datamodule: Created test dataset with {len(self.test)} samples")
        logging.info(f"Datamodule setup stage {stage} done.")

    def get_train_class_weights(self):

        num_queries = self.max_queries
        max_targets, d, no_class, overshoot = dataset_summaries(self.train.dataset, num_queries)
        d[self.num_classes] = no_class
        logging.info("Train Set")
        logging.info(f"Max targets {max_targets}")
        logging.info(f"Label Distribution {d}")
        logging.info(f"Overshoot {overshoot}")

        max_targets, d, no_class, overshoot = dataset_summaries(self.val.dataset, num_queries)
        d[self.num_classes] = no_class
        logging.info("Validation Set")
        logging.info(f"Max targets {max_targets}")
        logging.info(f"Label Distribution {d}")
        logging.info(f"Overshoot {overshoot}")

        max_targets, d, no_class, overshoot = dataset_summaries(self.test.dataset, num_queries)
        d[self.num_classes] = no_class
        logging.info("Validation Set")
        logging.info(f"Max targets {max_targets}")
        logging.info(f"Label Distribution {d}")
        logging.info(f"Overshoot {overshoot}")

    def train_dataloader(self):
        # return DataLoader(self.train, batch_size=self.batch_size, num_workers=4, sampler=self.train_sampler, collate_fn=collate_fn)
        return self.train

    def val_dataloader(self):
        return self.val

    def test_dataloader(self):
        return self.test
