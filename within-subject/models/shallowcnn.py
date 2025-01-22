# models/shallowcnn.py
# -*- coding: utf-8 -*-

"""ShallowCNN model definition.

A shallower CNN architecture inspired by FBCSP,
with a large temporal kernel, a spatial filter,
and log-type activation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ShallowCNN(nn.Module):
    """Shallow CNN architecture for EEG data.

    Args:
        n_chans (int): Number of EEG channels.
        n_times (int): Number of time samples.
        n_classes (int): Number of output classes.
        temporal_filters (int): Number of filters in the temporal conv.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        n_chans=72,
        n_times=250,
        n_classes=3,
        temporal_filters=40,
        dropout=0.5
    ):
        super().__init__()
        self.temp_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=temporal_filters,
                kernel_size=(1, 13),
                padding=(0, 6),
                bias=False
            ),
            nn.BatchNorm2d(temporal_filters)
        )
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(
                temporal_filters,
                temporal_filters,
                kernel_size=(n_chans, 1),
                groups=temporal_filters,
                bias=False
            ),
            nn.BatchNorm2d(temporal_filters)
        )
        self.pool = nn.AvgPool2d(kernel_size=(1, 35))
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
        out = torch.square(out)
        out = self.spatial_conv(out)
        out = F.logsigmoid(out)
        out = self.pool(out)
        out = self.dropout(out)
        out = out.squeeze(2).mean(dim=-1)
        out = self.classifier(out)
        return out
