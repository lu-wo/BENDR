"""Dataloader for the EMGRep project."""

import logging
from argparse import Namespace
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .EMGRepDataset import EMGRepDataset

import os
from typing import Any, Dict

import scipy.io as sio
import pytorch_lightning as pl


def collate_emg(batch):
    batch = list(zip(*batch))
    batch[0] = torch.stack(batch[0])
    batch[1] = torch.stack(batch[1])
    batch[2] = torch.stack(batch[2])
    batch[3] = torch.stack(batch[3])
    batch[5] = torch.stack(batch[5])
    # print(batch[0].size())
    return tuple(batch)


def load_mat(filename: str) -> Dict[str, Any]:
    """Load a .mat file.

    Args:
        filename (str): Path to the .mat file.

    Returns:
        Dict[str, Any]: Dictionary containing the variables in the .mat file.
    """
    return sio.loadmat(filename)


def get_recording(subject: int, day: int, time: int, data_path: str) -> Dict[str, Any]:
    """Load a recording from the dataset.

    Args:
        subject (int): Id of the subject (1-10).
        day (int): Day of the recording (1-5).
        time (int): Time of the recording (1-2).
        data_path (str): Path to the dataset.

    Raises:
        ValueError: If subject, day or time are not in the valid range.

    Returns:
        Dict[str, Any]: Dictionary containing the variables in the .mat file.
    """
    if subject not in range(1, 11):
        raise ValueError(f"Subject {subject} not in [1, 10]")

    if time not in range(1, 3):
        raise ValueError(f"Time {time} not in [1, 2]")

    if day in range(1, 6):
        fpath = os.path.join(data_path, f"S{subject}_D{day}_T{time}.mat")
    else:
        raise ValueError(f"Day {day} not in [1, 2, 3, 4, 5]")
    return load_mat(fpath)


class EMGRepDataModule(pl.LightningDataModule):
    """Dataloader class."""

    def __init__(
        self,
        data_path: Path,
        split_mode: str,
        n_subjects: int,
        n_days: int,
        n_times: int,
        val_idx: List[int],
        test_idx: List[int],
        positive_mode: str = "none",
        seq_len: int = 3000,
        seq_stride: int = 3000,
        block_len: int = 300,
        block_stride: int = 300,
        batch_size: int = 1,
        num_workers: int = 0,
    ):
        """Initialize the dataloader.

        Args:
            data_path (Path): Path to the data.
            train_data (List[Tuple[int, int, int]]): Data selected for training.
                Given as a list of tuples (subject, day, time).
            val_data (List[Tuple[int, int, int]], optional): Data selected for validation.
                Given as a list of tuples (subject, day, time). Defaults to []. If empty,
                no validation dataloader can be created.
            test_data (List[Tuple[int, int, int]], optional): Data selected for testing.
                Given as a list of tuples (subject, day, time). Defaults to []. If empty,
                no testing dataloader can be created.
            positive_mode (str, optional): Whether to use self or subject as positive class.
                Defaults to "none". Other options are: "session", "subject", "label".
            seq_len (int, optional): Length of the sequence. Defaults to 3000.
            seq_stride (int, optional): Stride of the sequence. Defaults to 3000.
            block_len (int, optional): Length of the block in sequence. Defaults to 300.
            block_stride (int, optional): Stride of the block in sequence. Defaults to 300.
            batch_size (int, optional): Batch size for the dataloader. Defaults to 1.
            num_workers (int, optional): Number of workers for the dataloader. Defaults to 0.
        """
        super().__init__()
        self.data_path = data_path
        self.split_mode = split_mode
        self.n_subjects = n_subjects
        self.n_days = n_days
        self.n_times = n_times
        self.val_idx = val_idx
        self.test_idx = test_idx
        train_split, val_split, test_split = get_split(
            split_mode, n_subjects, n_days, n_times, val_idx, test_idx
        )
        self.train_data = train_split
        self.val_data = val_split
        self.test_data = test_split
        self.positive_mode = positive_mode
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.block_len = block_len
        self.block_stride = block_stride
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _create_dataset(self, mode="train") -> EMGRepDataset:
        """Create the dataset.

        Args:
            mode (str, optional): Mode of the dataset. Defaults to "train".

        Returns:
            EMGRepDataset: Dataset.
        """
        if mode == "test":
            data = self.test_data

        elif mode == "train":
            data = self.train_data
        elif mode == "val":
            data = self.val_data

        logging.info(f"Loading {mode} dataset...")
        mat_files = [
            get_recording(
                subject=subject,
                day=day,
                time=time,
                data_path=str(self.data_path),
            )
            for subject, day, time in tqdm(
                data, desc=f"Loading {mode} dataset", ncols=100
            )
        ]
        return EMGRepDataset(
            mat_files=mat_files,
            positive_mode=self.positive_mode,
            seq_len=self.seq_len,
            seq_stride=self.seq_stride,
            block_len=self.block_len,
            block_stride=self.block_stride,
        )

    def get_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Get the dataloader.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation and test dataloader.
        """
        train_dataset = self._create_dataset(mode="train")
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_emg,
            drop_last=True,
        )
        val_loader = None
        if self.val_data:
            val_dataset = self._create_dataset(mode="val")
            val_loader = DataLoader(
                dataset=val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_emg,
                drop_last=True,
            )
        test_loader = None
        if self.test_data:
            test_dataset = self._create_dataset(mode="test")
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_emg,
                drop_last=True,
            )

        return train_loader, val_loader, test_loader

    def setup(self, stage: str = None):
        """Setup the dataloaders.

        Args:
            stage (str, optional): Stage of the dataloader. Defaults to None.
        """
        self.train, self.val, self.test = self.get_dataloaders()

    def train_dataloader(self):
        """Get the train dataloader.

        Returns:
            DataLoader: Train dataloader.
        """
        return self.train

    def val_dataloader(self):
        """Get the validation dataloader.

        Returns:
            DataLoader: Validation dataloader.
        """
        return self.val

    def test_dataloader(self):
        """Get the test dataloader.

        Returns:
            DataLoader: Test dataloader.
        """
        return self.test


def get_dataloader(
    args: Namespace, extract_rep_mode: bool = False
) -> Dict[str, DataLoader]:
    """Get the dataloaders.

    Args:
        args (Namespace): Command line arguments.
        extract_rep_mode (bool, optional): Whether to use block len as stride since we want to
        extract c_t for each block. Defaults to False.

    Returns:
        Dict[str, DataLoader]: Train, val, and test dataloaders.
    """
    train_split, val_split, test_split = get_split(args)

    dl = EMGRepDataloader(
        data_path=args.data,
        train_data=train_split,
        val_data=val_split,
        test_data=test_split,
        positive_mode=args.positive_mode,
        seq_len=args.seq_len,
        seq_stride=args.block_len if extract_rep_mode else args.seq_stride,
        block_len=args.block_len,
        block_stride=args.block_stride,
        batch_size=args.batch_size_cpc,
        num_workers=args.num_workers,
    )

    train_dl, val_dl, test_dl = dl.get_dataloaders()

    logging.info(f"Train samples: {len(train_dl.dataset)}")
    logging.info(f"Val   samples: {len(val_dl.dataset)}")
    logging.info(f"Test  samples: {len(test_dl.dataset)}")

    return {
        "train": train_dl,
        "val": val_dl,
        "test": test_dl,
    }


def get_split(
    split_mode: str,
    n_subjects: int,
    n_days: int,
    n_times: int,
    val_idx: List[int],
    test_idx: List[int],
) -> Tuple[
    List[Tuple[int, int, int]], List[Tuple[int, int, int]], List[Tuple[int, int, int]]
]:
    """Get the train, val, and test splits.

    Args:
        args (Namespace): Command line arguments.

    Raises:
        ValueError: If the positive mode is not recognized.

    Returns:
        List[Tuple[int, int, int]]: Train, val, and test splits.
    """
    subject_range = range(1, n_subjects + 1)
    day_range = range(1, n_days + 1)
    time_range = range(1, n_times + 1)

    if split_mode == "day":
        invalid_vals = [val for val in val_idx if val not in day_range]
        assert invalid_vals == [], f"Invalid val index: {invalid_vals}"
        invalid_test = [test for test in test_idx if test not in day_range]
        assert invalid_test == [], f"Invalid val index: {invalid_test}"
        train_split = [
            (subject, day, time)
            for subject in subject_range
            for day in day_range
            for time in time_range
            if day not in (val_idx + test_idx)
        ]
        val_split = [
            (subject, day, time)
            for subject in subject_range
            for day in val_idx
            for time in time_range
        ]
        test_split = [
            (subject, day, time)
            for subject in subject_range
            for day in test_idx
            for time in time_range
        ]
    elif split_mode == "subject":
        invalid_vals = [val for val in val_idx if val not in subject_range]
        assert invalid_vals == [], f"Invalid val index: {invalid_vals}"
        invalid_test = [test for test in test_idx if test not in subject_range]
        assert invalid_test == [], f"Invalid val index: {invalid_test}"

        train_split = [
            (subject, day, time)
            for subject in subject_range
            for day in day_range
            for time in time_range
            if (subject not in (val_idx + test_idx))
        ]
        val_split = [
            (subject, day, time)
            for subject in val_idx
            for day in day_range
            for time in time_range
        ]
        test_split = [
            (subject, day, time)
            for subject in test_idx
            for day in day_range
            for time in time_range
        ]
    else:
        raise ValueError(f"Invalid positive mode: {split_mode}")

    return train_split, val_split, test_split
