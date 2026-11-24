import math

import torch
import torch.nn as nn


def sinusoidal_encoding(length, dim, device):
    t = torch.arange(length, device=device, dtype=torch.float32)[:, None]
    freq = torch.exp(torch.arange(0, dim, 2, device=device).float() * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2], pe[:, 1::2] = torch.sin(t * freq), torch.cos(t * freq)
    return pe


class ObsDualContext(nn.Module):
    def __init__(self, in_dim, structure_mix, k, dim, n_heads, cycle, n_views, gate_hidden):
        super().__init__()
        self.k, self.dim, self.n_heads, self.cycle = k, dim, n_heads, cycle
        self.n_structures = structure_mix.shape[1]
        self.register_buffer("structure_mix", structure_mix, persistent=False)
        self.embed = nn.Sequential(nn.Linear(in_dim, dim), nn.LayerNorm(dim))
        self.qkv = nn.Linear(dim, 3 * dim)
        self.offset_bias = nn.Parameter(torch.zeros(4 * k + 1, n_heads))
        self.coverage_weight = nn.Parameter(torch.ones(n_heads))
        self.local_out = nn.Linear(dim, dim)
        self.structure_embedding = nn.Parameter(torch.randn(self.n_structures, dim) * 0.02)
        self.global_attention = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(3 * dim, dim), nn.GELU(), nn.Linear(dim, in_dim))
        self.gate = nn.Sequential(nn.Linear(n_views + self.n_structures, gate_hidden), nn.GELU(), nn.Linear(gate_hidden, 1))
        for layer in (self.fusion[-1], self.gate[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def window(self, frames, device):
        position = {int(t): i for i, t in enumerate(frames)}
        index, offset_row, valid = [], [], []
        for t in frames:
            for j in range(-self.k, self.k + 1):
                a = int(t) + j
                wrapped = not 0 <= a < self.cycle
                a %= self.cycle
                index.append(position.get(a, 0))
                valid.append(a in position)
                offset_row.append(2 * self.k + 1 + j + self.k - (j > 0) if wrapped else j + self.k)
        shape = (len(frames), 2 * self.k + 1)
        return (torch.tensor(index, device=device).view(shape), torch.tensor(offset_row, device=device).view(shape),
                torch.tensor(valid, device=device).view(shape))

    def local_context(self, e, coverage_total, frames):
        T, R, _ = e.shape
        head_dim = self.dim // self.n_heads
        q, k, v = self.qkv(e).view(T, R, 3, self.n_heads, head_dim).unbind(2)
        index, offset_row, valid = self.window(frames, e.device)
        logits = torch.einsum("trhc,twrhc->twrh", q, k[index]) / math.sqrt(head_dim)
        logits = logits + self.offset_bias[offset_row][:, :, None] + self.coverage_weight * torch.log(coverage_total[index] + 1e-3)[..., None]
        attention = logits.masked_fill(~valid[:, :, None, None], float("-inf")).softmax(1)
        return self.local_out(torch.einsum("twrh,twrhc->trhc", attention, v[index]).reshape(T, R, self.dim))

    def global_context(self, e, coverage_total, frames):
        T, R, _ = e.shape
        weights = coverage_total[..., None] * self.structure_mix
        tokens = torch.einsum("trs,trd->tsd", weights, e) / (weights.sum(1)[..., None] + 1e-3)
        pe = sinusoidal_encoding(self.cycle, self.dim, e.device)[[int(t) for t in frames]]
        tokens = (tokens + pe[:, None] + self.structure_embedding).reshape(1, T * self.n_structures, self.dim)
        queries = (e + pe[:, None] + self.structure_mix @ self.structure_embedding).reshape(1, T * R, self.dim)
        return self.global_attention(queries, tokens, tokens, need_weights=False)[0].view(T, R, self.dim)

    def forward(self, h, coverage, frames):
        e = self.embed(h)
        coverage_total = coverage.sum(-1).clamp(max=1.0)
        local = self.local_context(e, coverage_total, frames)
        global_ = self.global_context(e, coverage_total, frames)
        gate = torch.sigmoid(self.gate(torch.cat([coverage, self.structure_mix.expand(h.shape[0], -1, -1)], -1)))
        return h + gate * self.fusion(torch.cat([e, local, global_], -1))
