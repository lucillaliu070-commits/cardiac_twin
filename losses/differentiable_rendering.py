import torch
import torch.nn as nn
from pytorch3d.ops import knn_points


class DifferentiableRenderingLoss(nn.Module):
    def __init__(self, structure_vertices, tau, alpha, mu, max_distance, min_weight, n_views=4):
        super().__init__()
        self.structure_vertices = [torch.as_tensor(v, dtype=torch.long) for v in structure_vertices]
        self.tau, self.alpha, self.mu = tau, alpha, mu
        self.max_distance, self.min_weight, self.n_views = max_distance, min_weight, n_views
        self.band = tau + 7.0 / alpha

    def plane_association(self, distance):
        return 1.0 - torch.exp(-self.mu * torch.sigmoid(self.alpha * (self.tau - distance)))

    def render_structure(self, verts, planes, vertices, reference, n_reference):
        x = verts[planes["frame"]][:, vertices.to(verts.device)]
        normal, centre = planes["normal"], planes["centre"]
        signed = ((x - centre[:, None]) * normal[:, None]).sum(-1)
        p = self.plane_association(signed.abs()).detach()
        q = x - signed[..., None] * normal[:, None]
        to_pixel = planes["to_pixel"]
        pixel = q @ to_pixel[:, :, :3].transpose(1, 2) + to_pixel[:, None, :, 3]
        inside = ((pixel >= -0.5) & (pixel <= planes["size"][:, None] - 0.5)).all(-1)
        with torch.no_grad():
            nearest = knn_points(q.detach(), reference, lengths2=n_reference.clamp(min=1), K=1).idx[..., 0]
        g = torch.gather(reference, 1, nearest[..., None].expand(-1, -1, 3))
        distance = torch.sqrt(((q - g) ** 2).sum(-1) + 1e-8).clamp(max=self.max_distance)
        weight = p * (inside & (signed.abs().detach() < self.band) & (n_reference[:, None] > 0))
        total = weight.sum(1)
        loss = (weight * distance).sum(1) / total.clamp(min=1e-12)
        valid = (n_reference > 0) & (total.detach() >= self.min_weight)
        return torch.where(valid, loss, torch.zeros_like(loss)), valid

    def forward(self, verts, planes, contours, n_contour_points):
        n_frames = verts.shape[0]
        pair_loss, pair_valid = [], []
        for s, vertices in enumerate(self.structure_vertices):
            loss, valid = self.render_structure(verts, planes, vertices, contours[:, s], n_contour_points[:, s])
            pair_loss.append(loss)
            pair_valid.append(valid)
        pair_loss, pair_valid = torch.stack(pair_loss, 1), torch.stack(pair_valid, 1)
        key = planes["frame"] * self.n_views + planes["view"]
        view_sum = verts.new_zeros(n_frames * self.n_views).index_add_(0, key, pair_loss.sum(1)).view(n_frames, self.n_views)
        view_count = verts.new_zeros(n_frames * self.n_views).index_add_(0, key, pair_valid.sum(1).float()).view(n_frames, self.n_views)
        has_view = view_count > 0
        view_mean = view_sum / view_count.clamp(min=1)
        frame_loss = (view_mean * has_view).sum(1) / has_view.sum(1).clamp(min=1)
        has_frame = has_view.any(1)
        return (frame_loss * has_frame).sum() / has_frame.sum().clamp(min=1)
