import torch
import torch.nn as nn


class GeoRefiner(nn.Module):
    def __init__(self, lookup, temporal_context, vertex_region, n_regions, region_dim, view_feature_dim, hidden,
                 region_embedding_dim, iteration_embedding_dim, n_iters, centre_unit, logscale_unit, n_views=4):
        super().__init__()
        self.lookup = lookup
        self.temporal_context = temporal_context
        self.n_regions, self.region_dim, self.n_iters = n_regions, region_dim, n_iters
        self.centre_unit, self.logscale_unit = centre_unit, logscale_unit
        vertex_region = torch.as_tensor(vertex_region, dtype=torch.long)
        self.register_buffer("vertex_region", vertex_region, persistent=False)
        self.register_buffer("region_count", torch.bincount(vertex_region, minlength=n_regions).float(), persistent=False)
        self.register_buffer("latent_std", torch.ones(n_regions, region_dim))
        self.region_embedding = nn.Parameter(torch.randn(n_regions, region_embedding_dim) * 0.02)
        self.iteration_embedding = nn.Parameter(torch.randn(n_iters, iteration_embedding_dim) * 0.02)
        in_dim = n_views * (view_feature_dim + 1) + region_dim + region_embedding_dim + iteration_embedding_dim
        self.shape_head = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                        nn.Linear(hidden, region_dim))
        self.pose_head = nn.Sequential(nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, 4))
        for layer in (self.shape_head[-1], self.pose_head[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def region_evidence(self, mesh, centre, scale, geometry, images, features, faces):
        T = mesh.shape[0]
        parts, coverage = [], []
        for evidence, weight in self.lookup(mesh, centre, scale, geometry, images, features, faces):
            num = mesh.new_zeros(T, self.n_regions, evidence.shape[-1]).index_add_(1, self.vertex_region, evidence * weight[..., None])
            den = mesh.new_zeros(T, self.n_regions).index_add_(1, self.vertex_region, weight)
            view_coverage = (den / self.region_count)[..., None]
            parts += [num / (den[..., None] + 1e-3), view_coverage]
            coverage.append(view_coverage)
        return parts, torch.cat(coverage, -1)

    def forward(self, z, mesh, centre, scale, frames, geometry, images, features, faces, iteration):
        T = z.shape[0]
        parts, coverage = self.region_evidence(mesh, centre, scale, geometry, images, features, faces)
        h = torch.cat(parts + [z.detach().view(T, self.n_regions, self.region_dim) / self.latent_std,
                               self.region_embedding.expand(T, -1, -1),
                               self.iteration_embedding[iteration].expand(T, self.n_regions, -1)], -1)
        h = self.temporal_context(h, coverage, frames)
        delta_z = self.shape_head(h)
        total = coverage.sum(-1).clamp(max=1.0)
        pose = self.pose_head((h * total[..., None]).sum(1) / (total.sum(1, keepdim=True) + 1e-6))
        return z + (delta_z * self.latent_std).reshape(T, -1), pose, delta_z

    def refine(self, z, centre, scale, frames, geometry, images, features, decoder, faces):
        for iteration in range(self.n_iters):
            with torch.no_grad():
                mesh = decoder(z.detach())
            z, pose, _ = self.forward(z, mesh, centre.detach(), scale.detach(), frames, geometry, images, features, faces, iteration)
            centre = centre + self.centre_unit * pose[:, :3]
            scale = scale * torch.exp(self.logscale_unit * pose[:, 3])
        return z, centre, scale
