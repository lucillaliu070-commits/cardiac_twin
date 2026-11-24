import torch
import torch.nn as nn


class DecoderInjection(nn.Module):
    def __init__(self, lookup, level_channels, point_feature_dim, hidden):
        super().__init__()
        self.lookup = lookup
        self.adapters = nn.ModuleDict({str(level): nn.Sequential(nn.Linear(point_feature_dim + c, hidden), nn.GELU(), nn.Linear(hidden, c))
                                       for level, c in enumerate(level_channels, start=1)})
        for adapter in self.adapters.values():
            nn.init.zeros_(adapter[-1].weight)
            nn.init.zeros_(adapter[-1].bias)
        self.images, self.maps = None, None

    def prepare(self, images, features):
        self.images, self.maps = images, self.lookup.project_maps(features)

    def for_frame(self, t, centre, scale, geometry):
        centre, scale = centre.detach(), scale.detach()

        def inject(level, node_features, node_positions):
            evidence = self.lookup.point_features(node_positions.detach(), centre, scale, geometry, t, self.images, self.maps)
            return self.adapters[str(level)](torch.cat([evidence, node_features], -1))

        return inject
