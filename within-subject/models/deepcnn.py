# models/deepcnn.py
# -*- coding: utf-8 -*-

"""DeepCNN model definition.

A deeper CNN architecture for EEG classification, with
temporal and spatial convolutions followed by standard
convolution blocks.
"""

import torch
import torch.nn as nn


class DeepCNN(nn.Module):
    """Deeper CNN architecture for EEG data.

    Args:
        n_chans (int): Number of EEG channels.
        n_times (int): Number of time samples.
        n_classes (int): Number of output classes.
        temporal_filters (int): Number of filters in the first temporal conv.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        n_chans=72,
        n_times=250,
        n_classes=3,
        temporal_filters=15,
        dropout=0.5
    ):
        super().__init__()
        self.temp_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=temporal_filters,
                kernel_size=(1, 5),
                padding=(0, 2),
                bias=False
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU()
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=temporal_filters,
                out_channels=temporal_filters,
                kernel_size=(n_chans, 1),
                groups=temporal_filters,
                bias=False
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU()
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                temporal_filters,
                kernel_size=(1, 5),
                padding=(0, 2),
                bias=False
            ),
            nn.BatchNorm2d(temporal_filters),
            nn.ELU()
        )

        self.pool = nn.AvgPool2d(kernel_size=(1, 4))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(temporal_filters, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input of shape (B, 1, n_chans, n_times).

        Returns:
            Tensor: Logits of shape (B, n_classes).
        """
        out = self.temp_conv(x)
        out = self.spatial_conv(out)
        out = self.conv_block(out)
        out = self.pool(out) 
        out = out.squeeze(2).mean(dim=-1) 
        out = self.dropout(out)
        out = self.classifier(out)
        return out
