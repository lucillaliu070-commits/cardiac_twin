import torch
import torch.nn as nn


class WHM4D(nn.Module):
    def __init__(self, image_encoder, multiview_fusion, latent_head, size_head, mesh_vae, geo_refiner, decoder_injection):
        super().__init__()
        self.image_encoder = image_encoder
        self.multiview_fusion = multiview_fusion
        self.latent_head = latent_head
        self.size_head = size_head
        self.mesh_vae = mesh_vae
        self.geo_refiner = geo_refiner
        self.decoder_injection = decoder_injection

    def decode(self, z, centre, scale, geometry):
        meshes = []
        for t in range(z.shape[0]):
            inject = self.decoder_injection.for_frame(t, centre[t], scale[t], geometry)
            mesh = self.mesh_vae.decoder(z[t:t + 1], inject)[0]
            meshes.append(mesh * scale[t] + centre[t])
        return torch.stack(meshes)

    def to_world(self, mesh, cardiac_frame):
        return mesh @ cardiac_frame[:, :3, :3].transpose(1, 2) + cardiac_frame[:, None, :3, 3]

    def forward(self, images, geometry, frames, faces):
        bottleneck, features = self.image_encoder(images)
        fused = self.multiview_fusion(bottleneck)
        z, _ = self.latent_head(fused)
        centre, scale = self.size_head(fused, frames)
        z, centre, scale = self.geo_refiner.refine(z, centre, scale, frames, geometry, images, features,
                                                   self.mesh_vae.decoder, faces)
        self.decoder_injection.prepare(images, features)
        mesh = self.decode(z, centre, scale, geometry)
        return self.to_world(mesh, geometry["cardiac_frame"]), mesh
