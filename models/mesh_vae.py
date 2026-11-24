import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.graph_layers import ChebConv, ExphormerAttention, Pool, exphormer_batch


class GraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.conv = ChebConv(in_channels, out_channels, K)
        self.norm = nn.InstanceNorm1d(out_channels)

    def forward(self, x, edges):
        return F.relu(self.norm(self.conv(x.transpose(1, 2), edges))).transpose(1, 2)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, K, n_heads):
        super().__init__()
        self.gcn = ChebConv(in_channels, out_channels, K)
        self.gcn_norm = nn.InstanceNorm1d(out_channels)
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.attention = ExphormerAttention(out_channels, out_channels, n_heads, False)
        self.attention_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x, edges):
        p = self.proj(x)
        local = p + F.relu(self.gcn_norm(self.gcn(x, edges)))
        B, C, N = p.shape
        e = self.attention(exphormer_batch(p, edges)).view(B, N, C).permute(0, 2, 1)
        return local + self.attention_norm(p + e)


class DecoderLevel(nn.Module):
    def __init__(self, in_channels, out_channels, K, n_heads):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.gcn = GraphConv(in_channels, out_channels, K)
        self.attention = ExphormerAttention(out_channels, out_channels, n_heads, False)
        self.attention_norm = nn.BatchNorm1d(out_channels)
        self.position_head = ChebConv(out_channels, 3, 1, bias=False)

    def forward(self, x, edges):
        p = self.proj(x.transpose(1, 2))
        B, C, N = p.shape
        local = p.transpose(1, 2) + self.gcn(x, edges)
        e = self.attention(exphormer_batch(p, edges)).view(B, N, C).permute(0, 2, 1)
        h = local + self.attention_norm(p + e).transpose(1, 2)
        positions = self.position_head(h.transpose(1, 2), edges).transpose(1, 2)
        return h, positions


class RegionMeanPool(nn.Module):
    def __init__(self, region_ids):
        super().__init__()
        region_ids = torch.as_tensor(region_ids, dtype=torch.long)
        n_regions, n_nodes = int(region_ids.max()) + 1, region_ids.numel()
        M = torch.zeros(n_regions, n_nodes)
        M[region_ids, torch.arange(n_nodes)] = 1.0
        self.register_buffer("M", M / M.sum(1, keepdim=True), persistent=False)

    def forward(self, x):
        return torch.einsum("bcn,gn->bgc", x, self.M)


class RegionLinear(nn.Module):
    def __init__(self, n_regions, in_channels, region_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_regions, in_channels, region_dim).uniform_(-in_channels ** -0.5, in_channels ** -0.5))
        self.bias = nn.Parameter(torch.empty(n_regions, region_dim).uniform_(-in_channels ** -0.5, in_channels ** -0.5))

    def forward(self, x):
        return (torch.einsum("bgc,gcd->bgd", x, self.weight) + self.bias).flatten(1)


class RegionNodeDecoder(nn.Module):
    def __init__(self, region_ids, region_dim, out_channels):
        super().__init__()
        self.register_buffer("region_ids", torch.as_tensor(region_ids, dtype=torch.long))
        self.n_regions = int(self.region_ids.max()) + 1
        self.region_dim, self.out_channels = region_dim, out_channels
        self.weight = nn.Parameter(torch.randn(self.region_ids.numel(), region_dim, out_channels) * 0.1)
        self.bias = nn.Parameter(torch.zeros(self.region_ids.numel(), out_channels))

    def forward(self, z):
        B = z.shape[0]
        z_node = z.view(B, self.n_regions, self.region_dim)[:, self.region_ids]
        return (torch.einsum("bnd,ndc->bnc", z_node, self.weight) + self.bias).flatten(1)


class LoRARegionDecoder(nn.Module):
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        in_dim = base.n_regions * base.region_dim
        out_dim = base.region_ids.numel() * base.out_channels
        self.scaling = alpha / float(rank)
        self.A = nn.Parameter(torch.zeros(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(rank, out_dim))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        for p in self.base.parameters():
            p.requires_grad_(False)

    def forward(self, z):
        return self.base(z) + (z @ self.A.t()) @ self.B * self.scaling


class MeshEncoder(nn.Module):
    def __init__(self, filters, K, n_heads, adjacency, downsample, region_ids, region_dim):
        super().__init__()
        channels = [3] + list(filters)
        self.blocks = nn.ModuleList([EncoderBlock(channels[i], channels[i + 1], K, n_heads) for i in range(len(filters))])
        self.bottleneck = ChebConv(filters[-1], filters[-1], K)
        self.bottleneck_norm = nn.InstanceNorm1d(filters[-1])
        self.pool = Pool()
        self.adjacency, self.downsample = adjacency, downsample
        n_regions = int(torch.as_tensor(region_ids).max()) + 1
        self.region_pool = RegionMeanPool(region_ids)
        self.mu = RegionLinear(n_regions, filters[-1], region_dim)
        self.logvar = RegionLinear(n_regions, filters[-1], region_dim)

    def forward(self, verts):
        x = verts.transpose(1, 2)
        for level, block in enumerate(self.blocks):
            x = block(x, self.adjacency[level]._indices())
            x = self.pool(x.transpose(1, 2), self.downsample[level]).transpose(1, 2)
        edges = self.adjacency[len(self.blocks)]._indices()
        x = x + F.relu(self.bottleneck_norm(self.bottleneck(x, edges)))
        pooled = self.region_pool(x)
        return self.mu(pooled), self.logvar(pooled)


class MeshDecoder(nn.Module):
    def __init__(self, filters, K, n_heads, adjacency, upsample, region_ids, region_dim):
        super().__init__()
        f = filters
        self.n_coarse, self.coarse_channels = len(region_ids), f[4]
        self.latent_to_nodes = RegionNodeDecoder(region_ids, region_dim, f[4])
        widths = [(f[4], f[3]), (f[3] + 3, f[2]), (f[2] + 3, f[1]), (f[1] + 3, f[0]), (f[0] + 3, f[0])]
        self.levels = nn.ModuleList([DecoderLevel(i, o, K, n_heads) for i, o in widths])
        self.refine = GraphConv(f[0] + 3, f[0], K)
        self.out = ChebConv(f[0], 3, K, bias=False)
        self.unpool = Pool()
        self.adjacency, self.upsample = adjacency, upsample

    def forward(self, z, injection=None):
        B = z.shape[0]
        x = F.relu(self.latent_to_nodes(z)).view(B, self.n_coarse, self.coarse_channels)
        n_levels = len(self.levels)
        for level, block in enumerate(self.levels, start=1):
            h, positions = block(x, self.adjacency[n_levels + 1 - level]._indices())
            if injection is not None:
                h = h + injection(level, h, positions)
            x = self.unpool(torch.cat([h, positions], -1), self.upsample[n_levels - level])
        edges = self.adjacency[0]._indices()
        x = self.refine(x, edges)
        return self.out(x.transpose(1, 2), edges).transpose(1, 2)


class MeshVAE(nn.Module):
    def __init__(self, filters, K, n_heads, adjacency, downsample, upsample, region_ids, region_dim):
        super().__init__()
        self.encoder = MeshEncoder(filters, K, n_heads, adjacency, downsample, region_ids, region_dim)
        self.decoder = MeshDecoder(filters, K, n_heads, adjacency, upsample, region_ids, region_dim)

    def forward(self, verts):
        mu, logvar = self.encoder(verts)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) if self.training else mu
        return self.decoder(z), mu, logvar

    def freeze_with_lora(self, rank, alpha):
        self.requires_grad_(False)
        self.decoder.latent_to_nodes = LoRARegionDecoder(self.decoder.latent_to_nodes, rank, alpha)
