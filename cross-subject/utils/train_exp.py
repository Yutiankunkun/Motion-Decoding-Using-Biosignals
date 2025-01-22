# utils/train_exp.py
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from .merge_data import merged_dataset 
from .train_eval import fit_and_evaluate_datasets


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_multi(
    model_cls,
    subject_ids: list[str],
    base_dir: str,
    seq_length: int,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    transform,
    num_channels: int,
    num_classes: int,
    save_dir: str,
    num_runs: int = 1,
    seeds: list[int] = None
):
    """
    Leave-One-Subject-Out training repeated multiple times with different seeds.

    For each subject in subject_ids, use that subject as validation,
    merge all other subjects as training data. Optionally repeat multiple times.

    Args:
        model_cls (type): model class, e.g. EEGNet
        subject_ids (list[str]): e.g. ['subject0','subject1','subject2',...]
        base_dir (str): top-level directory containing subject subfolders
        seq_length (int): EEG sequence length
        num_epochs (int): training epochs
        batch_size (int): batch size
        device (torch.device): GPU or CPU
        transform (callable): data augmentation
        num_channels (int): EEG channels
        num_classes (int): number of classes
        save_dir (str): directory to save trained models
        num_runs (int): how many times to repeat for each subject
        seeds (list[int]): list of seeds. If None, seeds = range(num_runs)

    Returns:
        results (dict): a dict subject -> list of run_info
                        run_info includes best_val_acc, seed, etc.
    """

    if seeds is None:
        seeds = list(range(num_runs))

    os.makedirs(save_dir, exist_ok=True)

    results = {}

    for test_subj in subject_ids:
        print(f"\n=== Leave {test_subj} Out ===")
        train_subjs = [s for s in subject_ids if s != test_subj]

        train_data = merged_dataset(
            subject_ids=train_subjs,
            base_dir=base_dir,
            seq_length=seq_length,
            is_train=True,
            transform=transform
        )

        val_data = merged_dataset(
            subject_ids=[test_subj],
            base_dir=base_dir,
            seq_length=seq_length,
            is_train=False,  
            transform=transform
        )

        subj_results = []
        for run_idx in range(num_runs):
            seed = seeds[run_idx] * 10 + 10
            print(f"\n  Run {run_idx+1}/{num_runs}, seed={seed}")
            set_seed(seed)

            model, history, best_val_acc = fit_and_evaluate_datasets(
                model_cls=model_cls,
                train_dataset=train_data,
                val_dataset=val_data,
                device=device,
                num_channels=num_channels,
                num_classes=num_classes,
                seq_length=seq_length,
                num_epochs=num_epochs,
                batch_size=batch_size,
                lr=1e-3
            )

            model_filename = f"{test_subj}_seed{seed}.pth"
            save_path = os.path.join(save_dir, model_filename)
            torch.save(model.state_dict(), save_path)

            run_info = {
                'run_idx': run_idx,
                'seed': seed,
                'best_val_acc': best_val_acc,
                'model_path': save_path
            }
            subj_results.append(run_info)

        results[test_subj] = subj_results

    return results

def compute(results, key='best_val_acc'):

    values = np.array([r[key] for subject in results.values() for r in subject])
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) 
    
    return {
        'mean': mean_val,
        'std': std_val,
    }