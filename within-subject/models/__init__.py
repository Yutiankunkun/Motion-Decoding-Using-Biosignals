# models/__init__.py

"""Init file for the 'models' package.

This file centralizes imports of various EEG models defined
in separate modules within this package.
"""

from .eegnet import EEGNet
from .deepcnn import DeepCNN
from .shallowcnn import ShallowCNN
from .fbcnet import FBCNet
from .eegconformer import EEGConformer
from .esnnet import ESNNet

__all__ = [
    "EEGNet",
    "DeepCNN",
    "ShallowCNN",
    "FBCNet",
    "EEGConformer",
    "ESNNet"
]
