from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from lrnnx.models.lti.lru import LRU


class LayerNormFeature(nn.Module):
    """Layer normalization over feature (channel) dimension."""

    def __init__(self, num_features: int):
        super().__init__()
        # nn.LayerNorm normalizes over the last dimension (C) for (B, T, C)
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        return self.norm(x)


class DownPool1D(nn.Module):
    """
    1D downsampling: stride-k Conv1d that doubles channels.
    Expects (B, C, T) in, returns (B, 2C, T/k) when T is divisible by k.
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
        # x: (B, C, T)
        return self.conv(x)


class UpPool1D(nn.Module):
    """
    1D upsampling: ConvTranspose1d with stride-k that halves channels.
    Expects (B, C, T) in, returns (B, C/2, T*k) (with a fixed causal shift).
    """

    def __init__(
        self, in_channels: int, upsample_factor: int = 2, causal: bool = True
    ):
        super().__init__()
        self.causal = causal
        self.factor = upsample_factor
        self.deconv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=in_channels // 2,
            kernel_size=upsample_factor,
            stride=upsample_factor,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.deconv(x)
        if self.causal:
            # Fixed causal shift to align with downsampling convention
            x = F.pad(x, (1, 0))[:, :, :-1]
        return x


class LRUBlock(nn.Module):
    """
    Wrapper around lrnnx LRU for 1D tensors in (B, T, C) format.
    LRU expects (B, L, H) == (B, T, C).
    """

    def __init__(self, N: int, H: int):
        super().__init__()
        self.lru_layer = LRU(N=N, H=H)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) where C == H
        return self.lru_layer(x)


class LRUUnet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: List[int] = [8, 16],
        resample_factors: List[int] = [4, 2],
        pre_conv: bool = False,  # unused
        causal: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        if len(channels) != len(resample_factors):
            raise ValueError(
                f"Length of channels ({len(channels)}) must match "
                f"length of resample_factors ({len(resample_factors)})"
            )

        self.depth = len(channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = [in_channels] + channels
        self.resample_factors = resample_factors
        self.pre_conv = pre_conv
        self.causal = causal

        # Total downsampling factor (product of all r's)
        self.total_downsample = 1
        for r in resample_factors:
            self.total_downsample *= r

        # Encoder
        self.down_ssms = nn.ModuleList()
        enc_out_chs = []  # channels after each DownPool

        cur_ch = in_channels  # start with input channels
        for c_target, r in zip(channels, resample_factors):
            self.down_ssms.append(
                nn.Sequential(
                    # (B, T, cur_ch) -> (B, T, cur_ch)
                    self.build_lru_block(cur_ch, use_activation=True),
                    # (B, cur_ch, T) -> (B, 2*cur_ch, T/r)
                    DownPool1D(cur_ch, downsample_factor=r),
                )
            )
            cur_ch = cur_ch * 2
            enc_out_chs.append(cur_ch)

        encoder_out_channels = enc_out_chs

        # Bottleneck
        final_encoder_channels = encoder_out_channels[-1]
        self.hid_ssms = nn.Sequential(
            self.build_lru_block(final_encoder_channels, use_activation=True),
            self.build_lru_block(final_encoder_channels, use_activation=True),
        )

        # Decoder
        self.up_ssms = nn.ModuleList()
        cur_ch = final_encoder_channels
        for enc_out_c, r in zip(
            encoder_out_channels[::-1], resample_factors[::-1]
        ):
            self.up_ssms.append(
                nn.Sequential(
                    # (B, cur_ch, T) -> (B, cur_ch//2, T*r)
                    UpPool1D(cur_ch, upsample_factor=r, causal=causal),
                    # LRU on (B, T, cur_ch//2)
                    self.build_lru_block(cur_ch // 2, use_activation=True),
                )
            )
            cur_ch = cur_ch // 2

        # Final processing
        self.last_ssms = nn.Sequential(
            self.build_lru_block(in_channels, use_activation=True),
            self.build_lru_block(in_channels, use_activation=False),
        )

        if in_channels != out_channels:
            self.final_proj = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
                dtype=dtype,
            )
        else:
            self.final_proj = nn.Identity()

    def build_lru_block(
        self, in_channels: int, use_activation: bool = False
    ) -> nn.Sequential:
        """
        LRU block that accepts (B, T, C=in_channels) and preserves C.
        """
        seq = nn.Sequential()
        seq.append(LRUBlock(N=in_channels, H=in_channels))

        if use_activation:
            if in_channels > 1:
                seq.append(LayerNormFeature(in_channels))
            seq.append(nn.SiLU())

        return seq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:  (B, Cin, T)
        Output: (B, Cout, T)
        """
        # Pad input so T is divisible by total_downsample
        T = x.shape[2]  # (B, C_in, T)
        original_length = T
        pad_amount = (
            self.total_downsample - (T % self.total_downsample)
        ) % self.total_downsample
        if pad_amount > 0:
            x = F.pad(x, (0, pad_amount))  # (B, C_in, T + pad_amount)

        x = x.permute(0, 2, 1)  # (B, T, C_in)

        skips = []

        # Encoder
        for down_seq in self.down_ssms:
            skips.append(x)
            x = down_seq[0](x)  # (B, T, C)

            # go to (B, C, T) for Conv1d
            x_conv = x.permute(0, 2, 1)  # (B, C, T)
            x_conv = down_seq[1](x_conv)  # DownPool1D: (B, 2C, T/r)
            x = x_conv.permute(0, 2, 1)  # back to (B, T, 2C)

        # Bottleneck
        x = self.hid_ssms(x)  # (B, T, C_enc_last)

        # Decoder
        for up_seq, skip in zip(self.up_ssms, skips[::-1]):
            # go to (B, C, T) for ConvTranspose1d
            x_conv = x.permute(0, 2, 1)  # (B, C, T)
            x_conv = up_seq[0](x_conv)  # UpPool1D: (B, C/2, T*r)
            x = x_conv.permute(0, 2, 1)  # (B, T, C/2)

            x = x + skip
            x = up_seq[1](x)  # (B, T, C)

        # Final processing
        x = self.last_ssms(x)  # (B, T, C_in)

        # Back to (B, C, T)
        x = x.permute(0, 2, 1)  # (B, C_in, T)
        x = self.final_proj(x)  # (B, C_out, T)

        # Crop back to original (un-padded) length
        x = x[:, :, :original_length]  # (B, C_out, original_T)

        return x
