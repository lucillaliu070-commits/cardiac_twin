import torch
import torch.nn as nn
import torch.nn.functional as F

VIEWS = ("2ch", "3ch", "4ch", "sax")


class PlaneLookup(nn.Module):
    def __init__(self, lookup_vertices, offsets, feature_offsets, encoder_stages, stage_channels, feature_channels,
                 sigma, image_size, n_slices, n_lax_planes=3, n_planes=16):
        super().__init__()
        self.register_buffer("vertices", torch.as_tensor(lookup_vertices, dtype=torch.long), persistent=False)
        self.register_buffer("offsets", torch.as_tensor(offsets, dtype=torch.float32), persistent=False)
        self.center_offset = list(offsets).index(0.0)
        self.feature_offsets = [list(offsets).index(float(o)) for o in feature_offsets]
        self.encoder_stages = tuple(encoder_stages)
        self.sigma, self.image_size, self.n_slices = sigma, image_size, n_slices
        self.n_lax_planes, self.n_planes = n_lax_planes, n_planes
        self.proj = nn.ModuleDict({f"{kind}{stage}": (nn.Conv3d if kind == "sax" else nn.Conv2d)(c, feature_channels, 1)
                                   for kind in ("lax", "sax") for stage, c in zip(encoder_stages, stage_channels)})

    @staticmethod
    def vertex_normals(V, faces):
        f = V[:, faces]
        fn = torch.cross(f[:, :, 1] - f[:, :, 0], f[:, :, 2] - f[:, :, 0], dim=-1)
        n = torch.zeros_like(V).index_add_(1, faces.reshape(-1), fn.repeat_interleave(3, dim=1))
        return F.normalize(n, dim=-1)

    def view_slots(self, v):
        return slice(v, v + 1) if v < self.n_lax_planes else slice(self.n_lax_planes, self.n_planes)

    def grid(self, rcd, slice_index=None):
        g = [(rcd[..., 1] + 0.5) / self.image_size * 2 - 1, (rcd[..., 0] + 0.5) / self.image_size * 2 - 1]
        if slice_index is not None:
            g.append((slice_index + 0.5) / self.n_slices * 2 - 1)
        return torch.stack(g, -1)

    def project_maps(self, features):
        return {view: [self.proj[f"{'sax' if view == 'sax' else 'lax'}{s}"](features[view][s]) for s in self.encoder_stages]
                for view in VIEWS}

    def plane_weight(self, rcd, valid):
        inside = ((rcd[..., 0] > -0.5) & (rcd[..., 0] < self.image_size - 0.5) & (rcd[..., 1] > -0.5)
                  & (rcd[..., 1] < self.image_size - 0.5)).float()
        return torch.exp(-rcd[..., 2] ** 2 / (2 * self.sigma ** 2)) * valid[..., None] * inside

    def forward(self, mesh, centre, scale, geometry, images, features, faces):
        T = mesh.shape[0]
        A, G = geometry["cardiac_frame"], geometry["plane_to_pixel"]
        valid, slice_index = geometry["plane_valid"], geometry["sax_slice"]
        points = mesh[:, self.vertices] * scale.view(T, 1, 1) + centre.view(T, 1, 3)
        normals = F.normalize(self.vertex_normals(mesh, faces)[:, self.vertices] @ A[:, :3, :3].transpose(1, 2), dim=-1)
        world = (points @ A[:, :3, :3].transpose(1, 2) + A[:, None, :3, 3])[:, :, None] + self.offsets.view(1, 1, -1, 1) * normals[:, :, None]
        rcd = torch.einsum("tpij,tsoj->tpsoi", G[..., :3], world) + G[:, :, None, None, :, 3]
        maps = self.project_maps(features)
        out = []
        for v, view in enumerate(VIEWS):
            sl = self.view_slots(v)
            r = rcd[:, sl]
            w = self.plane_weight(r[:, :, :, self.center_offset], valid[:, sl])
            if v < self.n_lax_planes:
                profile = F.grid_sample(images[view], self.grid(r[:, 0]), align_corners=False)
                g = self.grid(r[:, 0][:, :, self.feature_offsets])
                sampled = [F.grid_sample(m, g, align_corners=False) for m in maps[view]]
                weight = w[:, 0]
            else:
                g = self.grid(r, slice_index[:, sl].view(T, -1, 1, 1).expand_as(r[..., 0]))
                wn = (w / (w.sum(1, keepdim=True) + 1e-6))[:, None, :, :, None]
                profile = (F.grid_sample(images[view], g, align_corners=False) * wn).sum(2)
                g = g[:, :, :, self.feature_offsets]
                sampled = [(F.grid_sample(m, g, align_corners=False) * wn).sum(2) for m in maps[view]]
                weight = w.sum(1).clamp(max=1.0)
            evidence = torch.cat([profile.permute(0, 2, 1, 3).flatten(2)] + [s.permute(0, 2, 1, 3).flatten(2) for s in sampled], -1)
            out.append((evidence, weight))
        return out

    def point_features(self, points, centre, scale, geometry, t, images, maps):
        A = geometry["cardiac_frame"][t:t + 1]
        G = geometry["plane_to_pixel"][t:t + 1]
        valid, slice_index = geometry["plane_valid"][t:t + 1], geometry["sax_slice"][t:t + 1]
        P = points * scale.view(1, 1, 1) + centre.view(1, 1, 3)
        world = P @ A[:, :3, :3].transpose(1, 2) + A[:, None, :3, 3]
        rcd = torch.einsum("tpij,tnj->tpni", G[..., :3], world) + G[:, :, None, :, 3]
        out = []
        for v, view in enumerate(VIEWS):
            sl = self.view_slots(v)
            r = rcd[:, sl]
            w = self.plane_weight(r, valid[:, sl])
            frame_maps = [images[view][t:t + 1]] + [m[t:t + 1] for m in maps[view]]
            if v < self.n_lax_planes:
                g = self.grid(r[:, 0])[:, :, None]
                values = [F.grid_sample(m, g, align_corners=False)[..., 0] for m in frame_maps]
                weight = w[:, 0]
            else:
                g = self.grid(r, slice_index[:, sl].view(1, -1, 1).expand_as(r[..., 0]))[:, :, :, None]
                wn = (w / (w.sum(1, keepdim=True) + 1e-6))[:, None]
                values = [(F.grid_sample(m, g, align_corners=False)[..., 0] * wn).sum(2) for m in frame_maps]
                weight = w.sum(1).clamp(max=1.0)
            f = torch.cat(values, 1).permute(0, 2, 1)
            out.append(torch.cat([f * weight[..., None], weight[..., None]], -1))
        return torch.cat(out, -1)
