# models/eegnet.py
# -*- coding: utf-8 -*-

"""EEGNet model definition.

Implements a lightweight CNN architecture with separate
temporal and spatial filtering (Depthwise/Separable Convolution)
designed for EEG data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SeparableConv2d(nn.Module):
    """Depthwise-Separable 2D Convolution.

    Splits standard convolution into depthwise + pointwise steps
    to reduce parameters.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple): Convolutional kernel size (h, w).
        bias (bool): Whether to use a bias term in conv layers.
    """

    def __init__(self, in_channels, out_channels, kernel_size, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            bias=bias
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input of shape (B, C, H, W).

        Returns:
            Tensor: Output of shape (B, out_channels, H, W').
        """
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


class EEGNet(nn.Module):
    """EEGNet architecture for EEG signal classification.

    Reference: "EEGNet: A Compact Convolutional Network for EEG-based
    Brain-Computer Interfaces" by Lawhern et al.

    Args:
        n_chans (int): Number of EEG channels (electrodes).
        n_times (int): Number of time samples per trial.
        n_classes (int): Number of output classes.
        temporal_filters (int): Number of temporal filters.
        depthwise_filters (int): Depth multiplier in the Depthwise step.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        n_chans=72,
        n_times=250,
        n_classes=3,
        temporal_filters=4,
        depthwise_filters=8,
        dropout=0.5
    ):
        super().__init__()
        self.temporal_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=temporal_filters,
                kernel_size=(1, 125),
                padding=(0, 62),
                bias=False
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU()
        )

        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=temporal_filters,
                out_channels=temporal_filters * depthwise_filters,
                kernel_size=(n_chans, 1),
                groups=temporal_filters,
                bias=False
            ),
            nn.BatchNorm2d(temporal_filters * depthwise_filters),
            nn.ELU()
        )

        self.separable_conv = nn.Sequential(
            SeparableConv2d(
                in_channels=temporal_filters * depthwise_filters,
                out_channels=temporal_filters * depthwise_filters,
                kernel_size=(1, 16),
                bias=False
            ),
            nn.BatchNorm2d(temporal_filters * depthwise_filters),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(
            temporal_filters * depthwise_filters,
            n_classes
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): EEG input of shape (B, 1, n_chans, n_times).

        Returns:
            Tensor: Logits of shape (B, n_classes).
        """
        out = self.temporal_conv(x)  
        out = self.depthwise(out)   
        out = self.separable_conv(out) 
        out = out.squeeze(2) 
        out = out.mean(dim=-1) 
        out = self.classifier(out)
        return out