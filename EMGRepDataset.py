"""Dataset for the EMGRep project."""

from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset
from dataset.dataset_whole import create_annotations


"""
we will always use positive mode
"""


class EMGRepDataset(Dataset):
    """Dataset for the EMGRep project."""

    def __init__(
        self,
        mat_files: List[Dict[str, Any]],
        positive_mode: str = "none",
        seq_len: int = 3000,
        seq_stride: int = 3000,
        block_len: int = 300,
        block_stride: int = 300,
    ) -> None:
        """Initialize the dataset.

        Args:
            mat_files (List[Dict[str, Any]]): List containing the mat files.
            positive_mode (str, optional): Whether to additionally sample
                acc and object data
            seq_len (int, optional): Length of the sequence. Defaults to 3000.
            seq_stride (int, optional): Stride of the sequence. Defaults to 3000.
            block_len (int, optional): Length of the block in sequence. Defaults to 300.
            block_stride (int, optional): Stride of the block in sequence. Defaults to 300.
        """
        super().__init__()

        self.positive_mode = positive_mode
        self.seq_len = seq_len
        self.seq_stride = seq_stride
        self.block_len = block_len
        self.block_stride = block_stride

        self.num_classes = 11

        assert self.positive_mode in {
            "none",
            "objects",
            "acc",
        }, f"Positive mode {self.positive_mode} must be 'none', 'objects', 'acc'."

        self.emg, self.stimulus, self.info, self.object, self.acc = self._load_data(
            mat_files
        )

        self.rng = np.random.default_rng(seed=42)

    def _load_data(
        self, mat_files: List[Dict[str, Any]]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Creates sequences from the data.

        Args:
            mat_files (List[Dict[str, Any]]): List containing the mat files.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: EMG, stimulus and info.
        """
        emg = []
        stimulus = []
        object = []
        masks = []
        info = []
        acc_data = []
        for mat_file in mat_files:
            signal = mat_file["emg"]
            label = mat_file["restimulus"]
            reobject = mat_file["reobject"]
            acc = mat_file["acc"]

            label = label - 1  # se
            # indices = label == 0
            # label[indices] == self.num_classes

            idx = 0
            while idx + self.seq_len <= signal.shape[0]:
                emg.append(signal[idx : idx + self.seq_len])
                stimulus.append(label[idx : idx + self.seq_len])
                if self.positive_mode == "objects":
                    object.append(reobject[idx : idx + self.seq_len])
                elif self.positive_mode == "acc":
                    acc_data.append(acc[idx : idx + self.seq_len])

                info.append(
                    np.array(
                        [
                            mat_file["subj"][0, 0],
                            mat_file["daytesting"][0, 0],
                            mat_file["time"][0, 0],
                            int(stimulus[-1][self.seq_len // 2]),
                        ]
                    )
                )
                idx += self.seq_stride

        emg = np.stack(emg)
        stimulus = np.stack(stimulus)
        info = np.stack(info)
        if self.positive_mode == "objects":
            object = np.stack(object)
            unique = np.unique(object)
            print(f"Unique labels: {unique}")
            map_dict = dict(zip(unique, range(len(unique))))
            object = np.vectorize(map_dict.get)(object)
            unique = np.unique(object)
            print(f"New unique labels: {unique}")

        else:
            object = None
        if self.positive_mode == "acc":
            acc_data = np.stack(acc_data)
        else:
            acc_data = None

        return emg, stimulus, info, object, acc_data

    def _seq_to_blocks(self, signal) -> np.ndarray:
        """Converts a sequence to blocks.

        Args:
            signal (np.ndarray): input signal.

        Returns:
            np.ndarray: blocks.
        """
        blocks = []

        idx = 0
        while idx + self.block_len <= signal.shape[0]:
            blocks.append(signal[idx : idx + self.block_len])
            idx += self.block_stride

        return np.stack(blocks)

    # TODO: needs to be corrected
    def _sample_positive_seq(
        self, info: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Samples a positive sequence based on the positive mode.

        Args:
            info (np.ndarray): Information of the sequence.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: EMG, stimulus, and info of positive sample.

        Notes:
            The positive mode controls a subset of data the positive samples are drawn from. It can
            be one of the following:
                - subject: Positive samples must come from the same subject.
                - session: Positive samples must come from the same session.
                - label: No additional constraint.
        """
        assert self.positive_mode in {
            "subject",
            "session",
            "label",
        }, "Positive mode must be 'subject', 'session' or 'label'."

        if self.positive_mode == "subject":
            positive_mode_condition = np.all(
                self.info[:, [0, 3]] == info[[0, 3]], axis=1
            )

        if self.positive_mode == "session":
            positive_mode_condition = np.all(self.info == info, axis=1)

        if self.positive_mode == "label":
            positive_mode_condition = self.info[:, -1] == info[-1]

        positive_indices = positive_mode_condition.nonzero()[0]
        positive_idx = self.rng.choice(positive_indices)

        return (
            self.emg[positive_idx],
            self.stimulus[positive_idx],
            self.info[positive_idx],
        )

    def __len__(self) -> int:
        """Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return self.emg.shape[0]

    def __getitem__(
        self, idx: int
    ) -> Union[
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
        Tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ],
    ]:
        """Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: EMG, stimulus and info. Note that EMG
            is of shape (batch_size, 2 (or 1), num_blocks, block_size, num_sensors) and stimulus is
            of shape (batch_size, 2 (or 1), num_blocks, block_size, 1). Info is of shape
            (batch_size, 2 (or 1), 4).
        """
        # emg_blocks = self._seq_to_blocks(self.emg[idx])
        # stimulus_blocks = self._seq_to_blocks(self.stimulus[idx])
        emg_blocks = self.emg[idx]
        emg_blocks = torch.from_numpy(emg_blocks)
        stimulus_blocks = self.stimulus[idx]

        stimulus_blocks = torch.from_numpy(stimulus_blocks)
        mask = stimulus_blocks == -1
        boxes, regions = create_annotations(stimulus_blocks)

        info = self.info[idx]
        info = torch.from_numpy(info)

        if self.positive_mode == "objects":
            emg = emg_blocks
            stimulus = stimulus_blocks
            object_blocks = self.object[idx]
            object_blocks = torch.from_numpy(object_blocks)
            return (
                emg.float(),  # seq, 16
                stimulus.float(),  # seq
                info.float(),
                regions,
                boxes,
                mask,
                object_blocks.long(),  # seq
            )
        elif self.positive_mode == "acc":
            emg = emg_blocks
            stimulus = stimulus_blocks
            acc_data = self.acc[idx]
            acc_data = torch.from_numpy(acc_data)
            return (
                emg.float(),
                stimulus.float(),
                info.float(),
                regions,
                boxes,
                mask,
                acc_data.float(),  # seq, 48
            )
        elif self.positive_mode == "none":
            # emg = emg_blocks.unsqueeze(0)
            # stimulus = stimulus_blocks.unsqueeze(0)
            # info = info.unsqueeze(0)
            emg = emg_blocks
            stimulus = stimulus_blocks

        else:
            raise NotImplementedError(
                f"Positive mode {self.positive_mode} is not implemented yet."
            )
            # positive_emg, positive_stimulus, positive_info = self._sample_positive_seq(info)
            # positive_emg_blocks = self.(positive_emg)
            # positive_stimulus_blocks = self._seq_to_blocks(positive_stimulus)
            #
            # emg = np.stack([emg_blocks, positive_emg_blocks])
            # stimulus = np.stack([stimulus_blocks, positive_stimulus_blocks])
            # info = np.stack([info, positive_info])

        return (emg.float(), stimulus.float(), info.float(), regions, boxes, mask)
