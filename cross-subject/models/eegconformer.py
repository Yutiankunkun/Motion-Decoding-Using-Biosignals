# models/eegconformer.py
# -*- coding: utf-8 -*-

"""EEGConformer model definition.

A model combining CNN and Transformer blocks
for capturing local and global dependencies in EEG signals.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """A simple convolution block for local feature extraction.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (tuple): Size of the temporal kernel.
    """

    def __init__(self, in_channels, out_channels, kernel_size=(1, 25)):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(0, kernel_size[1] // 2),
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input of shape (B, C, H, W).

        Returns:
            Tensor: Output of shape (B, C', H, W').
        """
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class TransformerEncoder(nn.Module):
    """A simplified Transformer encoder block.

    Args:
        d_model (int): Dimensionality of the embedding.
        n_head (int): Number of attention heads.
        dim_feedforward (int): Dimensionality of the feedforward layer.
        dropout (float): Dropout rate.
    """

    def __init__(self, d_model=40, n_head=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_head, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Shape (B, seq_len, d_model).

        Returns:
            Tensor: Encoded sequence of shape (B, seq_len, d_model).
        """
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_out))

        ff_out = self.linear2(self.dropout(self.act(self.linear1(x))))
        x = self.norm2(x + self.dropout(ff_out))
        return x


class EEGConformer(nn.Module):
    """EEGConformer: CNN + Transformer for EEG classification.

    Args:
        n_chans (int): Number of EEG channels.
        n_times (int): Number of time samples.
        n_classes (int): Number of output classes.
        hidden_dim (int): Dimensionality for the CNN/Transformer latent space.
        dropout (float): Dropout rate.
        n_heads (int): Number of attention heads in the Transformer block.
    """

    def __init__(
        self,
        n_chans=72,
        n_times=250,
        n_classes=3,
        hidden_dim=20,
        dropout=0.5,
        n_heads=4
    ):
        super().__init__()
        self.temp_conv = ConvBlock(1, hidden_dim, kernel_size=(1, 25))
        self.spatial_conv = nn.Conv2d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=(n_chans, 1),
            groups=hidden_dim,
            bias=False
        )
        self.bn_spatial = nn.BatchNorm2d(hidden_dim)
        self.act_spatial = nn.ELU()

        self.transformer_encoder = TransformerEncoder(
            d_model=hidden_dim,
            n_head=n_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=0.1
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (Tensor): Input of shape (B, 1, n_chans, n_times).

        Returns:
            Tensor: Logits of shape (B, n_classes).
        """
        x = self.temp_conv(x)  
        x = self.spatial_conv(x)  
        x = self.bn_spatial(x)
        x = self.act_spatial(x)
        x = x.squeeze(2).permute(0, 2, 1)  
        x = self.transformer_encoder(x)
        x = x.mean(dim=1) 
        x = self.dropout(x)
        x = self.classifier(x)
        return x
