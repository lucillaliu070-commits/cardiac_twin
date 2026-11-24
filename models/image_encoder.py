import torch
import torch.nn as nn


class ImageEncoder(nn.Module):
    def __init__(self, lax_encoder, sax_encoder, n_trainable_stages, lookup_stages):
        super().__init__()
        self.lax = lax_encoder
        self.sax = sax_encoder
        self.n_trainable_stages = n_trainable_stages
        self.lookup_stages = tuple(lookup_stages)
        for encoder in (self.lax, self.sax):
            encoder.requires_grad_(False)
            for stage in encoder.stages[len(encoder.stages) - n_trainable_stages:]:
                stage.requires_grad_(True)

    def encode(self, encoder, x):
        n_frozen = max(0, len(encoder.stages) - self.n_trainable_stages)
        kept = {}
        with torch.no_grad():
            for i, stage in enumerate(encoder.stages[:n_frozen]):
                x = stage(x)
                if i in self.lookup_stages:
                    kept[i] = x
        for i, stage in enumerate(encoder.stages[n_frozen:], start=n_frozen):
            x = stage(x)
            if i in self.lookup_stages:
                kept[i] = x
        return x, kept

    def forward(self, images):
        bottleneck, features = {}, {}
        for view, x in images.items():
            encoder = self.sax if view == "sax" else self.lax
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=x.is_cuda):
                h, kept = self.encode(encoder, x)
            bottleneck[view] = h.float()
            features[view] = {i: f.float() for i, f in kept.items()}
        return bottleneck, features
