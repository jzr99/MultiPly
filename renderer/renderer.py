import torch
import torch.nn as nn
import numpy as np
import os

from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    PointLights,
    TexturesVertex,
    Textures,
    Materials,
)
from pytorch3d.structures import Meshes

materials = Materials(ambient_color=((0.7, 0.7, 0.7),), device='cuda')

from typing import NamedTuple, Sequence

class SMPLBlendPar(NamedTuple):
    sigma: float = 1e-9 # 0 sys.float_info.epsilon
    gamma: float = 1e-4
    background_color: Sequence = (1.0, 1.0, 1.0)

class Renderer(nn.Module):
    def __init__(self, batch_size, image_size, f=50, principal_point=((0.0, 0.0),), R=None, t=None):

        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.f = f
        self.principal_point = principal_point

        self.blend_params = SMPLBlendPar()

        if R is None and t is None: 
            self.cam_R = torch.from_numpy(np.array([[-1., 0., 0.],
                                                    [0., -1., 0.],
                                                    [0., 0., 1.]])).cuda().float().unsqueeze(0)

            
            self.cam_T = torch.zeros((1,3)).cuda().float()

            # no adaption to PyTorch3D needed
        else:
            # using the 'cam_poses' from 3DPW
            self.cam_R = R.detach().clone()
            self.cam_T = t.detach().clone()     

            # coordinate system adaption to PyTorch3D
            self.cam_R[:, :2, :] *= -1.0
            self.cam_T[:, :2] *= -1.0
            self.cam_R = torch.transpose(self.cam_R,1,2)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        torch.cuda.set_device(self.device)

        self.cameras = SfMPerspectiveCameras(focal_length=self.f, principal_point=self.principal_point, R=self.cam_R, T=self.cam_T, device=self.device)

        self.raster_settings = RasterizationSettings(image_size=image_size,faces_per_pixel=10, blur_radius=0)

        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.mask_shader = SoftSilhouetteShader(blend_params=self.blend_params)
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 2.0]])

        self.vis_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=SoftPhongShader(
                                                                                            device=self.device, 
                                                                                            cameras=self.cameras,
                                                                                            lights=lights,
                                                                                            materials=materials
                                                                                           ))
        self.mask_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.mask_shader)

        

    def forward(self, vertices, faces, mode='a'):   

        if 'a' in mode:

            verts_rgb = torch.ones_like(vertices).cuda()
            self.mask_texture = TexturesVertex(verts_features=verts_rgb)

            mask_mesh = Meshes(verts=vertices.float(),faces=faces.expand(self.batch_size,-1,-1),textures=self.mask_texture).cuda()

            mask = self.mask_renderer(mask_mesh)[...,3] 

            return mask, mask_mesh

        elif 'vis' in mode:
            verts_rgb = torch.ones_like(vertices).cuda()
            self.mask_texture = TexturesVertex(verts_features=verts_rgb)
            vis_mesh = Meshes(verts=vertices,faces=faces.expand(self.batch_size,-1,-1),textures=self.mask_texture).cuda()
            rendered_img = self.vis_renderer(vis_mesh)
            return rendered_img, vis_mesh

        else:
            raise NotImplementedError('%s mode not supported by SMPLRenderer yet.' % mode)