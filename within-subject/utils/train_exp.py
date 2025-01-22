# utils/train_exp.py
# -*- coding: utf-8 -*-

"""Train models with different seeds and save trained models.

Set seed randomly and try to run experiment multiple times.
"""

import os
import random
import numpy as np
import torch
from utils.train_eval import fit_and_evaluate


def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train_multi(
    model_cls,
    train_dir,
    val_dir,
    seq_length,
    num_epochs,
    batch_size,
    device,
    transform,
    num_channels,
    num_classes,
    save_dir,
    num_runs=5,
    seeds=None
):
    """
    Repeat the training procedure multiple times with different random seeds.

    Args:
        model_cls (type): The model class to instantiate.
        train_dir (str): Training data folder.
        val_dir (str): Validation data folder.
        seq_length (int): Sequence length for cropping.
        num_epochs (int): Number of epochs to train.
        batch_size (int): Batch size.
        device (torch.device): CPU or CUDA device.
        transform (callable): Transform function for data augmentation.
        num_channels (int): EEG channels.
        num_classes (int): Number of classes.
        save_dir (str): Directory to save the trained models.
        num_runs (int): How many times to repeat the experiment.
        seeds (list or None): List of seeds. If None, seeds = range(num_runs).

    Returns:
        results (list): Each element is a dict with info about that run, e.g.
                        {
                          'run_idx': int,
                          'seed': int,
                          'best_val_acc': float,
                          'model_path': str
                        }
    """


    if seeds is None:
        seeds = list(range(num_runs))

    results = []

    for run_idx in range(num_runs):
        seed = seeds[run_idx] * 10 + 10
        print(f"\n=== Run {run_idx+1}/{num_runs} | Seed: {seed} ===")

        set_seed(seed)

        model, history, best_val_acc = fit_and_evaluate(
            model_cls=model_cls,
            train_dir=train_dir,
            val_dir=val_dir,
            seq_length=seq_length,
            num_epochs=num_epochs,
            batch_size=batch_size,
            device=device,
            transform=transform,
            num_channels=num_channels,
            num_classes=num_classes
        )

        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f"{model_cls.__name__}_seed{seed}.pth")
        torch.save(model.state_dict(), model_path)

        run_info = {
            'run_idx': run_idx,
            'seed': seed,
            'best_val_acc': best_val_acc,
            'model_path': model_path
        }
        results.append(run_info)

    return results

def compute(results, key='best_val_acc'):

    values = np.array([r[key] for r in results])
    mean_val = np.mean(values)
    std_val = np.std(values, ddof=1) 
    
    return {
        'mean': mean_val,
        'std': std_val,
    }