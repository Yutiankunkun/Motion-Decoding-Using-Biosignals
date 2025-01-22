# utils/__init__.py
# -*- coding: utf-8 -*-

"""Utilities package initialization.

This file centralizes imports from various utility modules
(data_io, dataset, train_eval) so that they can be more easily
accessed by outside scripts or notebooks.

Example usage in other scripts:
    from utils import make_data, SeqDataset, train_evaluate
"""

from .data_io import (
    load_data,
    make_data,
    mne_data,
    visualize
)

from .dataset import (
    SeqDataset,
    standardization,
    preprocess,
    add_noise,
    transform_func
)

from .train_eval import (
    train_epoch,
    validate_epoch,
    final_evaluate,
    fit_and_evaluate,
    plot_history
)

from .train_exp import(
    train_multi,
    compute
)

__all__ = [
    # data_io
    "load_data",
    "make_data",
    "mne_data",
    "visualize",

    # dataset
    "SeqDataset",
    "standardization",
    "preprocess",
    "add_noise",
    "transform_func",

    # train_eval
    "train_epoch",
    "validate_epoch",
    "final_evaluate",
    "fit_and_evaluate",
    "plot_history",

    # train_exp
    "train_multi",
    "compute"
]
