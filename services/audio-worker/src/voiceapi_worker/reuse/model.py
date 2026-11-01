# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1
#
# Model definition of NVIDIA NemotronLabs RE-USE (SEMamba, time-d4 variant), vendored from
# huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE at revision 31f82f2efad1095034492f236e8dac9239ba713b
# (models/generator_SEMamba_time_d4.py, models/mamba_block2_SEMamba.py, models/codec_module_time_d4.py)
# so the worker never executes code fetched at runtime. Changes from upstream:
# * ``mamba_ssm.Mamba`` is replaced by ``Mamba`` below: identical parameters and inference math,
#   calling the selective-scan kernel directly (see scan.py), so state dicts load unchanged.
# * The PyTorchModelHubMixin base is dropped; weights load through ``load_reuse``.
# See LICENSE.OpenMDW-1.1 in this directory.

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

from voiceapi_worker.reuse.scan import selective_scan


class Mamba(nn.Module):
    """mamba_ssm v1 ``Mamba`` (inference path), parameter-compatible with its checkpoints."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner,
                                padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.A_log = nn.Parameter(torch.empty(self.d_inner, d_state))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        length = hidden_states.shape[1]
        xz = rearrange(self.in_proj(hidden_states), "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        x = F.silu(self.conv1d(x)[..., :length])
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj.weight @ dt.t(), "d (b l) -> b d l", l=length)
        B = rearrange(B, "(b l) n -> b n l", l=length).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=length).contiguous()
        A = -torch.exp(self.A_log.float())
        y = selective_scan(x, dt, A, B, C, self.D.float(), z, self.dt_proj.bias.float())
        return self.out_proj(rearrange(y, "b d l -> b l d"))


def get_padding_2d(kernel_size, dilation=(1, 1)):
    return (int((kernel_size[0] * dilation[0] - dilation[0]) / 2),
            int((kernel_size[1] * dilation[1] - dilation[1]) / 2))


class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super().__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.0)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        return out.contiguous().view((batch_size, nchannels // self.r, H, -1))


class DenseBlock(nn.Module):
    def __init__(self, cfg, kernel_size=(3, 3), depth=4):
        super().__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList()
        hid = cfg["model_cfg"]["hid_feature"]
        for i in range(depth):
            dil = 2**i
            self.dense_block.append(
                nn.Sequential(
                    nn.Conv2d(hid * (i + 1), hid, kernel_size, dilation=(dil, 1),
                              padding=get_padding_2d(kernel_size, (dil, 1))),
                    nn.InstanceNorm2d(hid, affine=True),
                    nn.PReLU(hid),
                )
            )

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x


class DenseEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hid = cfg["model_cfg"]["hid_feature"]
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(cfg["model_cfg"]["input_channel"], hid, (1, 1)),
            nn.InstanceNorm2d(hid, affine=True),
            nn.PReLU(hid),
        )
        self.dense_block = DenseBlock(cfg, depth=4)
        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(hid, hid, (1, 3), stride=(4, 2)),
            nn.InstanceNorm2d(hid, affine=True),
            nn.PReLU(hid),
        )

    def forward(self, x):
        return self.dense_conv_2(self.dense_block(self.dense_conv_1(x)))


class _UpsamplingDecoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hid = cfg["model_cfg"]["hid_feature"]
        self.dense_block = DenseBlock(cfg, depth=4)
        self.up_conv1 = nn.Sequential(SPConvTranspose2d(hid, hid, (1, 3), 2), nn.InstanceNorm2d(hid, affine=True),
                                      nn.PReLU(hid))
        self.up_conv2 = nn.Sequential(SPConvTranspose2d(hid, hid, (1, 3), 4), nn.InstanceNorm2d(hid, affine=True),
                                      nn.PReLU(hid))

    def _features(self, x):
        x = self.dense_block(x)
        x = self.up_conv1(x)
        return self.up_conv2(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)


class MagDecoder(_UpsamplingDecoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.final_conv = nn.Conv2d(cfg["model_cfg"]["hid_feature"], cfg["model_cfg"]["output_channel"], (1, 1))

    def forward(self, x):
        return self.final_conv(self._features(x))


class PhaseDecoder(_UpsamplingDecoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        hid = cfg["model_cfg"]["hid_feature"]
        out = cfg["model_cfg"]["output_channel"]
        self.phase_conv_r = nn.Conv2d(hid, out, (1, 1))
        self.phase_conv_i = nn.Conv2d(hid, out, (1, 1))

    def forward(self, x):
        x = self._features(x)
        return torch.atan2(self.phase_conv_i(x), self.phase_conv_r(x))


class MambaBlock(nn.Module):
    """Bidirectional Mamba with residuals, projection and LayerNorm."""

    def __init__(self, d_model, cfg):
        super().__init__()
        model_cfg = cfg["model_cfg"]
        kwargs = {"d_state": model_cfg["d_state"], "d_conv": model_cfg["d_conv"], "expand": model_cfg["expand"]}
        self.forward_blocks = Mamba(d_model=d_model, **kwargs)
        self.backward_blocks = Mamba(d_model=d_model, **kwargs)
        self.output_proj = nn.Linear(2 * d_model, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out_fw = self.forward_blocks(x) + x
        flipped = torch.flip(x, dims=[1])
        out_bw = torch.flip(self.backward_blocks(flipped) + flipped, dims=[1])
        return self.norm(self.output_proj(torch.cat([out_fw, out_bw], dim=-1)))


class TFMambaBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hid = cfg["model_cfg"]["hid_feature"]
        self.time_mamba = MambaBlock(d_model=hid, cfg=cfg)
        self.freq_mamba = MambaBlock(d_model=hid, cfg=cfg)

    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b * f, t, c)
        x = self.time_mamba(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b * t, f, c)
        x = self.freq_mamba(x) + x
        return x.view(b, t, f, c).permute(0, 3, 1, 2)


class SEMamba(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        blocks = cfg["model_cfg"]["num_tfmamba"] or 4
        self.dense_encoder = DenseEncoder(cfg)
        self.TSMamba = nn.ModuleList([TFMambaBlock(cfg) for _ in range(blocks)])
        self.mask_decoder = MagDecoder(cfg)
        self.phase_decoder = PhaseDecoder(cfg)

    def forward(self, noisy_mag, noisy_pha):
        noisy_mag = rearrange(noisy_mag, "b f t -> b t f").unsqueeze(1)
        noisy_pha = rearrange(noisy_pha, "b f t -> b t f").unsqueeze(1)
        x = torch.cat((noisy_mag, noisy_pha), dim=1)
        B, C, T, Fq = x.shape
        x = torch.cat((x, torch.zeros(B, C, T, 2, device=x.device)), dim=-1)
        x = torch.cat((x, torch.zeros(B, C, 2, Fq + 2, device=x.device)), dim=-2)
        x = self.dense_encoder(x)
        for block in self.TSMamba:
            x = block(x)
        denoised_mag = rearrange(self.mask_decoder(x), "b c t f -> b f t c").squeeze(-1)[:, :Fq, :T]
        denoised_pha = rearrange(self.phase_decoder(x), "b c t f -> b f t c").squeeze(-1)[:, :Fq, :T]
        denoised_com = torch.stack(
            (denoised_mag * torch.cos(denoised_pha), denoised_mag * torch.sin(denoised_pha)), dim=-1
        )
        return denoised_mag, denoised_pha, denoised_com


__all__ = ["Mamba", "SEMamba"]
