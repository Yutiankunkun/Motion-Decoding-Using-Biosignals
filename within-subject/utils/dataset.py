# utils/dataset.py
# -*- coding: utf-8 -*-

"""PyTorch Dataset definitions for EEG data.

Includes classes for reading preprocessed CSV segments,
applying transforms (e.g., random noise, time shift), etc.
"""

import os
import random
import glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def standardization(a: np.ndarray, axis=None, ddof=0) -> np.ndarray:
    """Standardize array along a specific axis.

    (a - mean) / std

    Args:
        a (np.ndarray): Input data.
        axis (int): Axis along which to compute mean/std.
        ddof (int): Delta Degrees of Freedom.

    Returns:
        np.ndarray: Standardized array (same shape as input).
    """
    a_mean = a.mean(axis=axis, keepdims=True)
    a_std = a.std(axis=axis, keepdims=True, ddof=ddof)
    a_std[a_std == 0] = 1
    return (a - a_mean) / a_std


def preprocess(df: pd.DataFrame) -> np.ndarray:
    """Convert DataFrame to numpy and standardize across channels or time.

    Args:
        df (pd.DataFrame): Input data with shape (time, channels).

    Returns:
        np.ndarray: 2D array of shape (channels, time).
    """
    mat = df.T.values
    mat = standardization(mat, axis=1)
    return mat


def add_noise(data: np.ndarray, noise_level=0.01) -> np.ndarray:
    """Add Gaussian noise to the data.

    Args:
        data (np.ndarray): Input array.
        noise_level (float): Std of the Gaussian noise.

    Returns:
        np.ndarray: Data with noise added.
    """
    noise = np.random.normal(0, noise_level, data.shape)
    data_noisy = data + noise
    return data_noisy.astype(np.float32)


def transform_func(array: np.ndarray, is_train: bool, seq_length: int) -> np.ndarray:
    """Example transform function to be used in the dataset.

    Args:
        array (np.ndarray): EEG array of shape (channels, n_times).
        is_train (bool): Flag if this is training mode (apply data aug).
        seq_length (int): Sequence length to crop or slice.

    Returns:
        np.ndarray: Transformed (channels, seq_length).
    """
    if is_train:
        _, n_times = array.shape
        start_idx = random.randint(0, n_times - seq_length)
        seq = array[:, start_idx:start_idx + seq_length]
        seq = add_noise(seq).astype(np.float32)
        if random.randint(0, 1):
            seq_r = seq[:, ::-1].copy()
            return seq_r
        return seq
    else:
        seq = array[:, :seq_length].astype(np.float32)
        return seq


class SeqDataset(Dataset):
    """Sequence Dataset that reads CSV files for each class.

    Args:
        root (str): Root directory containing subfolders of classes.
        seq_length (int): Desired sequence length per sample.
        is_train (bool): Whether training or not (affects transform).
        transform (callable): A function that transforms the (ch, time) array.
    """
    
    def __init__(self, root: str, seq_length: int, is_train: bool, transform=None):
        super().__init__()
        self.transform = transform
        self.seqs = []
        self.seq_labels = []
        self.class_names = sorted(os.listdir(root))
        self.class_names.sort()
        self.num_classes = len(self.class_names)
        self.seq_length = seq_length
        self.is_train = is_train

        for i, class_name in enumerate(self.class_names):
            temp = glob.glob(os.path.join(root, class_name, '*'))
            temp.sort()
            self.seq_labels.extend([i] * len(temp))
            for t in temp:
                df = pd.read_csv(t, header=None)
                tensor = preprocess(df)
                self.seqs.append(tensor)

    def __getitem__(self, index: int) -> dict:
        """Get item by index.

        Args:
            index (int): Sample index.

        Returns:
            dict: {
                'seq': Tensor of shape (channels, seq_length),
                'label': int label index
            }
        """
        seq = self.seqs[index]
        if self.transform is not None:
            seq = self.transform(seq, self.is_train, self.seq_length)
        seq = seq[None, :, :]
        return {
            'seq': seq,  # shape (channels, seq_length)
            'label': self.seq_labels[index]
        }

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.seqs)
