# models/fbcnet.py
# -*- coding: utf-8 -*-

"""FBCNet model definition.

A CNN architecture that leverages multi-band filtering
and spatial convolution for EEG classification.
"""

import torch
import torch.nn as nn


class FBCNet(nn.Module):
    """FBCNet architecture for multi-band EEG data.

    Args:
        n_chans (int): Number of EEG channels.
        n_times (int): Number of time samples.
        n_classes (int): Number of output classes.
        depthwise_filters (int): Number of depthwise filters.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        n_chans=72,
        n_times=250,
        n_classes=3,
        depthwise_filters=48,
        dropout=0.5
    ):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=depthwise_filters,
                kernel_size=(n_chans, 1),
                groups=1,  # or 1 if in_channels=1
                bias=False
            ),
            nn.BatchNorm2d(depthwise_filters),
            nn.ELU()
        )
        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(depthwise_filters, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input of shape (B, 1, n_chans, n_times).

        Returns:
            Tensor: Logits of shape (B, n_classes).
        """
        out = self.depthwise(x)
        out = self.pool(out)
        out = self.dropout(out)
        out = out.squeeze(2).mean(dim=-1)
        out = self.classifier(out)
        return out
