import torch
import torch.nn as nn
import torch.nn.functional as F


class PerViewAdaptor(nn.Module):
    def __init__(self, in_channels, out_channels, out_size):
        super().__init__()
        self.views = tuple(in_channels)
        self.out_size = out_size
        self.proj = nn.ModuleDict({view: nn.Conv2d(c, out_channels, 1) for view, c in in_channels.items()})
        self.norm = nn.ModuleDict({view: nn.GroupNorm(8, out_channels) for view in in_channels})

    def forward(self, views):
        out = []
        for view in self.views:
            x = F.interpolate(views[view], size=self.out_size, mode="bilinear", align_corners=False)
            out.append(self.norm[view](self.proj[view](x)))
        return torch.stack(out, 1)


class MultiViewFusion(nn.Module):
    def __init__(self, in_channels, out_channels, out_size):
        super().__init__()
        self.views = tuple(in_channels)
        self.adaptor = PerViewAdaptor(in_channels, out_channels, out_size)
        self.fuse = nn.Conv2d(out_channels * len(self.views), out_channels, 1)
        self.norm = nn.GroupNorm(8, out_channels)
        self.statistics = {}

    def set_input_statistics(self, statistics):
        self.statistics = statistics

    def forward(self, bottleneck):
        views = {}
        for view in self.views:
            x = bottleneck[view]
            if view in self.statistics:
                mean, std = self.statistics[view]
                x = (x - mean) / std
            views[view] = x.flatten(1, 2) if x.dim() == 5 else x
        x = self.adaptor(views)
        return self.norm(self.fuse(x.flatten(1, 2)))
