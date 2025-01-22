# utils/merge_data.py
# -*- coding: utf-8 -*-

"""Merge data from multiple subjects into one dataset by LOSO.

Leave one subject out(LOSO) and then merge the other datasets into one dataset.
"""

import os
from torch.utils.data import ConcatDataset


def merged_dataset(
    subject_ids: list[str],
    base_dir: str,
    seq_length: int,
    is_train: bool,
    transform
):
    """
    Merge data from multiple subjects into one dataset.

    Args:
        subject_ids (list[str]): e.g. ['subject0','subject1','subject2', ...]
        base_dir (str): Root directory containing subject subfolders.
        seq_length (int): Sequence length (cropping length).
        is_train (bool): If True, use training data subfolder; else val data subfolder.
        transform (callable): Data augmentation or transform function.

    Returns:
        ConcatDataset: A merged dataset of all specified subjects.
    """
    from .dataset import SeqDataset  

    all_datasets = []
    subfolder = "train" if is_train else "val"

    for sid in subject_ids:
        dir_path = os.path.join(base_dir, subfolder, sid)
        ds = SeqDataset(
            root=dir_path,
            seq_length=seq_length,
            is_train=is_train,
            transform=transform
        )
        all_datasets.append(ds)

    if len(all_datasets) == 1:
        return all_datasets[0]
    else:
        return ConcatDataset(all_datasets)
