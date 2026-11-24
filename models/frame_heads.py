import math

import torch
import torch.nn as nn


def sinusoidal_encoding(length, dim, device):
    t = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(t * freq), torch.cos(t * freq)
    return pe


class LatentHead(nn.Module):
    def __init__(self, in_channels, spatial_channels, pooled_size, hidden, n_heads, n_coefficients, latent_dim, dropout):
        super().__init__()
        self.hidden = hidden
        self.backbone = nn.Sequential(nn.Conv2d(in_channels, spatial_channels, 1), nn.AdaptiveAvgPool2d(pooled_size), nn.Flatten())
        self.proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(spatial_channels * pooled_size[0] * pooled_size[1], hidden),
                                  nn.Dropout(dropout))
        self.frame_block = nn.TransformerEncoderLayer(hidden, n_heads, 2 * hidden, dropout=dropout, batch_first=True)
        self.out = nn.Linear(hidden, n_coefficients)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("latent_mean", torch.zeros(latent_dim))
        self.register_buffer("basis", torch.zeros(n_coefficients, latent_dim))
        self.register_buffer("coefficient_std", torch.ones(n_coefficients))

    def forward(self, fused):
        n_frames = fused.shape[0]
        h = self.proj(self.backbone(fused)) + sinusoidal_encoding(n_frames, self.hidden, fused.device)
        frame_only = ~torch.eye(n_frames, dtype=torch.bool, device=fused.device)
        coefficients = self.out(self.frame_block(h[None], src_mask=frame_only))[0]
        latent = self.latent_mean + (coefficients * self.coefficient_std) @ self.basis
        return latent, coefficients


class SizeHead(nn.Module):
    def __init__(self, in_channels, hidden, n_phases):
        super().__init__()
        self.n_phases = n_phases
        self.norm = nn.LayerNorm(in_channels)
        self.mlp = nn.Sequential(nn.Linear(in_channels + 2, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.out = nn.Linear(hidden, 4)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)
        self.register_buffer("centre_prior", torch.zeros(n_phases, 3))
        self.register_buffer("scale_prior", torch.ones(n_phases))
        self.register_buffer("scale_unit", torch.ones(()))

    def forward(self, fused, phase):
        pooled = fused.mean(dim=(-2, -1))
        angle = 2 * math.pi * phase.float() / self.n_phases
        x = torch.cat([torch.sin(angle)[:, None], torch.cos(angle)[:, None], self.norm(pooled)], 1)
        o = self.out(self.mlp(x))
        centre = self.centre_prior[phase] + o[:, 1:] * self.scale_unit
        scale = self.scale_prior[phase] * torch.exp(o[:, 0])
        return centre, scale
