from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lrnnx.models.lti.lru import LRU


class LayerNormFeature(nn.Module):
    """
    Layer normalization over the feature (channel) dimension.
    
    :param num_features: Number of features (channels).
    :type num_features: int
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies normalization to input.

        :param x: Input of shape `(B, T, C)`.
        :type x: torch.Tensor
        :return: Normalized output.
        :rtype: torch.Tensor
        """
        return self.norm(x)


class DownPool1D(nn.Module):
    """
    1D downsampling: stride-k Conv1d that doubles channels.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param downsample_factor: Stride factor. Defaults to 2.
    :type downsample_factor: int, optional
    """

    def __init__(self, in_channels: int, downsample_factor: int = 2):
        super().__init__()
        self.factor = downsample_factor
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels * 2,
            kernel_size=downsample_factor,
            stride=downsample_factor,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Downsample input.

        :param x: Input of shape `(B, C, T)`.
        :type x: torch.Tensor
        :return: Downsampled output of shape `(B, 2C, T/k)`.
        :rtype: torch.Tensor
        """
        return self.conv(x)


class UpPool1D(nn.Module):
    """
    1D upsampling: stride-k ConvTranspose1d that halves channels.

    :param in_channels: Number of input channels.
    :type in_channels: int
    :param upsample_factor: Upsampling stride. Defaults to 2.
    :type upsample_factor: int, optional
    """

    def __init__(self, in_channels: int, upsample_factor: int = 2):
        super().__init__()
        self.factor = upsample_factor
        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=upsample_factor,
            stride=upsample_factor,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Upsample input.

        :param x: Input of shape `(B, C, T)`.
        :type x: torch.Tensor
        :return: Upsampled output of shape `(B, C/2, T*k)`.
        :rtype: torch.Tensor
        """
        return self.conv(x)


class LRU_UNet(nn.Module):
    """
    Linear Recurrent Unit (LRU) based U-Net for sequence tasks.

    :param d_model: Input feature dimension.
    :type d_model: int
    :param d_state: Hidden state dimension for the LRU layers.
    :type d_state: int
    :param n_layers: Number of downsampling/upsampling stages.
    :type n_layers: int
    :param downsample_factor: Factor for each stage. Defaults to 2.
    :type downsample_factor: int, optional
    """

    def __init__(
        self,
        d_model: int,
        d_state: int,
        n_layers: int,
        downsample_factor: int = 2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.total_downsample = downsample_factor**n_layers

        self.down_ssms = nn.ModuleList()
        curr_dim = d_model
        for _ in range(n_layers):
            self.down_ssms.append(
                nn.ModuleList(
                    [
                        LRU(curr_dim, d_state),
                        DownPool1D(curr_dim, downsample_factor),
                    ]
                )
            )
            curr_dim *= 2

        self.hid_ssms = LRU(curr_dim, d_state)

        self.up_ssms = nn.ModuleList()
        for _ in range(n_layers):
            self.up_ssms.append(
                nn.ModuleList(
                    [
                        UpPool1D(curr_dim, downsample_factor),
                        LRU(curr_dim // 2, d_state),
                    ]
                )
            )
            curr_dim //= 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the U-Net.

        :param x: Input sequence of shape `(B, C_in, T)`.
        :type x: torch.Tensor
        :return: Processed sequence of shape `(B, C_in, T)`.
        :rtype: torch.Tensor
        """
        # Handle padding for stride alignment
        T = x.shape[2]
        original_length = T
        pad_amount = (
            self.total_downsample - (T % self.total_downsample)
        ) % self.total_downsample
        if pad_amount > 0:
            x = F.pad(x, (0, pad_amount))

        x = x.permute(0, 2, 1)  # (B, T, C_in)
        skips = []

        # Encoder
        for lru_layer, pool_layer in self.down_ssms:
            skips.append(x)
            x = lru_layer(x)
            x_conv = x.permute(0, 2, 1)
            x_conv = pool_layer(x_conv)
            x = x_conv.permute(0, 2, 1)

        x = self.hid_ssms(x)

        # Decoder
        for pool_layer, lru_layer in self.up_ssms:
            x_conv = x.permute(0, 2, 1)
            x_conv = pool_layer(x_conv)
            x = x_conv.permute(0, 2, 1)
            skip = skips.pop()
            x = lru_layer(x + skip)

        x = x.permute(0, 2, 1)
        if pad_amount > 0:
            x = x[..., :original_length]

        return x