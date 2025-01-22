# models/esnnet.py
# -*- coding: utf-8 -*-

"""ESNNet model definition.

This module defines the ESNNet model for EEG classification, integrating
a simple CNN front-end with an Echo State Network (ESN) reservoir.

Model components:
1) CNN-based feature extractor for initial temporal and spatial filtering.
2) ESNBlock for temporal modeling using a fixed recurrent matrix.
3) A linear classifier to output final class logits.
"""

import torch
import torch.nn as nn


class ESNBlock(nn.Module):
    """
    A minimal Echo State Network (ESN) block.

    The recurrent weight matrix W is not trained; only the input-to-reservoir
    mapping (W_in) is trained. It returns the hidden states for each time step.

    Args:
        input_dim (int): Number of features for each time step (input size).
        hidden_dim (int): Size of the reservoir (hidden state dimension).
        spectral_radius (float): Controls the spectral radius of W for stability.
        leaky_rate (float): Leaking rate, 1.0 means no leaky integration.
        bias (bool): Whether to use bias in the input linear layer W_in.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        spectral_radius: float = 0.9,
        leaky_rate: float = 1.0,
        bias: bool = True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.leaky_rate = leaky_rate

        self.W_in = nn.Linear(input_dim, hidden_dim, bias=bias)

        W = torch.empty(hidden_dim, hidden_dim)
        nn.init.uniform_(W, a=-0.5, b=0.5)
        with torch.no_grad():
            W = W * spectral_radius / torch.linalg.norm(W, 2)
        self.W = nn.Parameter(W, requires_grad=False)

        self.activation = torch.tanh

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the ESN block.

        Args:
            x (torch.Tensor): Input sequence of shape (B, seq_len, input_dim).

        Returns:
            torch.Tensor: Hidden states at all time steps,
                          shape (B, seq_len, hidden_dim).
        """
        batch_size, seq_len, _ = x.size()
        h = x.new_zeros(batch_size, self.hidden_dim)

        outputs = []
        for t in range(seq_len):
            u = self.W_in(x[:, t, :]) + h @ self.W.T  
            # Leaky integration update
            h = (1 - self.leaky_rate) * h + self.leaky_rate * self.activation(u)
            outputs.append(h.unsqueeze(1)) 

        return torch.cat(outputs, dim=1)


class ESNNet(nn.Module):
    """
    ESNNet: A CNN front-end followed by an ESN for EEG classification.

    1) Temporal CNN for initial feature extraction.
    2) Spatial CNN for across-channels filtering.
    3) ESNBlock for temporal modeling of the extracted features.
    4) Global average pooling across time and a linear classifier.

    Args:
        n_chans (int): Number of EEG channels.
        n_times (int): Number of time samples in each EEG segment.
        n_classes (int): Number of output classes for classification.
        temporal_filters (int): Number of filters in the first CNN layer.
        reservoir_size (int): Size of the ESN reservoir (hidden state dimension).
        dropout (float): Dropout rate applied before the linear classifier.
    """
    def __init__(
        self,
        n_chans=72,
        n_times=250,
        n_classes=3,
        temporal_filters=25,
        reservoir_size=100,
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
 
        self.esn = ESNBlock(
            input_dim=temporal_filters,
            hidden_dim=reservoir_size,
            spectral_radius=0.99,
            leaky_rate=0.1
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(reservoir_size, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of ESNNet.

        Args:
            x (torch.Tensor): EEG input of shape (B, 1, n_chans, n_times).

        Returns:
            torch.Tensor: Logits of shape (B, n_classes).
        """
        out = self.temp_conv(x)  
        out = self.spatial_conv(out) 
        out = out.squeeze(2)
        out = out.permute(0, 2, 1)
        out = self.esn(out)
        out = out.mean(dim=1)  
        out = self.dropout(out)
        out = self.classifier(out)  
        return out