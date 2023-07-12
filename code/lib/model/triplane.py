import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from .grid import cuda_gridsample as cu
import math


def grid_sample(image, optical):
    N, C, IH, IW = image.shape
    _, H, W, _ = optical.shape

    ix = optical[..., 0]
    iy = optical[..., 1]

    ix = ((ix + 1) / 2) * (IW - 1);
    iy = ((iy + 1) / 2) * (IH - 1);
    with torch.no_grad():
        ix_nw = torch.floor(ix);
        iy_nw = torch.floor(iy);
        ix_ne = ix_nw + 1;
        iy_ne = iy_nw;
        ix_sw = ix_nw;
        iy_sw = iy_nw + 1;
        ix_se = ix_nw + 1;
        iy_se = iy_nw + 1;

    nw = (ix_se - ix) * (iy_se - iy)
    ne = (ix - ix_sw) * (iy_sw - iy)
    sw = (ix_ne - ix) * (iy - iy_ne)
    se = (ix - ix_nw) * (iy - iy_nw)

    with torch.no_grad():
        torch.clamp(ix_nw, 0, IW - 1, out=ix_nw)
        torch.clamp(iy_nw, 0, IH - 1, out=iy_nw)

        torch.clamp(ix_ne, 0, IW - 1, out=ix_ne)
        torch.clamp(iy_ne, 0, IH - 1, out=iy_ne)

        torch.clamp(ix_sw, 0, IW - 1, out=ix_sw)
        torch.clamp(iy_sw, 0, IH - 1, out=iy_sw)

        torch.clamp(ix_se, 0, IW - 1, out=ix_se)
        torch.clamp(iy_se, 0, IH - 1, out=iy_se)

    image = image.view(N, C, IH * IW)

    nw_val = torch.gather(image, 2, (iy_nw * IW + ix_nw).long().view(N, 1, H * W).repeat(1, C, 1))
    ne_val = torch.gather(image, 2, (iy_ne * IW + ix_ne).long().view(N, 1, H * W).repeat(1, C, 1))
    sw_val = torch.gather(image, 2, (iy_sw * IW + ix_sw).long().view(N, 1, H * W).repeat(1, C, 1))
    se_val = torch.gather(image, 2, (iy_se * IW + ix_se).long().view(N, 1, H * W).repeat(1, C, 1))

    out_val = (nw_val.view(N, C, H, W) * nw.view(N, 1, H, W) +
               ne_val.view(N, C, H, W) * ne.view(N, 1, H, W) +
               sw_val.view(N, C, H, W) * sw.view(N, 1, H, W) +
               se_val.view(N, C, H, W) * se.view(N, 1, H, W))

    return out_val


EPS = 1e-3

class TriPlane(nn.Module):
    def __init__(self, number_person, features=64, resX=128, resY=128, resZ=128):
        super().__init__()
        assert resX == resY == resZ, "resX, resY, resZ must be the same"
        self.encoder = nn.Embedding(number_person, 3 * features * resX * resY)
        self.plane_xy_list = nn.ParameterList()
        self.plane_yz_list = nn.ParameterList()
        self.plane_xz_list = nn.ParameterList()
        for p in range(number_person):
            person_id_tensor = torch.from_numpy(np.array([p])).long()
            person_encoding = self.encoder(person_id_tensor.to(self.encoder.weight.device)).reshape(3, features, resX, resY)
            self.plane_xy_list.append(person_encoding[[0]])
            self.plane_yz_list.append(person_encoding[[1]])
            self.plane_xz_list.append(person_encoding[[2]])
        # self.plane_xy = nn.Parameter(torch.randn(1, features, resX, resY))
        # self.plane_xz = nn.Parameter(torch.randn(1, features, resX, resZ))
        # self.plane_yz = nn.Parameter(torch.randn(1, features, resY, resZ))
        self.dim = features
        self.n_input_dims = 3
        # self.n_output_dims = 3 * features
        self.n_output_dims = features

    def forward(self, x, person_id):
        # assert x.max() <= 1 + EPS and x.min() >= -EPS, f"x must be in [0, 1], got {x.min()} and {x.max()}"
        # valid input [-1, 1]
        plane_xy = self.plane_xy_list[person_id]
        plane_xz = self.plane_xz_list[person_id]
        plane_yz = self.plane_yz_list[person_id]
        # x = x * 2 - 1
        shape = x.shape
        coords = x.reshape(1, -1, 1, 3)
        # align_corners=True ==> the extrema (-1 and 1) considered as the center of the corner pixels
        # F.grid_sample: [1, C, H, W], [1, N, 1, 2] -> [1, C, N, 1]
        # padding_mode='zeros' ==> the value of the pixels outside the grid is considered as 0
        # feat_xy = F.grid_sample(plane_xy, coords[..., [0, 1]], align_corners=True, padding_mode="border")[0, :, :, 0].transpose(0, 1)
        # feat_xz = F.grid_sample(plane_xz, coords[..., [0, 2]], align_corners=True, padding_mode="border")[0, :, :, 0].transpose(0, 1)
        # feat_yz = F.grid_sample(plane_yz, coords[..., [1, 2]], align_corners=True, padding_mode="border")[0, :, :, 0].transpose(0, 1)
        # feat_xy = cu.grid_sample_2d(plane_xy, coords[..., [0, 1]], align_corners=True, padding_mode="border")[0, :, :,
        #           0].transpose(0, 1)
        # feat_xz = cu.grid_sample_2d(plane_xz, coords[..., [0, 2]], align_corners=True, padding_mode="border")[0, :, :,
        #           0].transpose(0, 1)
        # feat_yz = cu.grid_sample_2d(plane_yz, coords[..., [1, 2]], align_corners=True, padding_mode="border")[0, :, :,
        #           0].transpose(0, 1)
        feat_xy = grid_sample(plane_xy, coords[..., [0, 1]],)[0, :, :, 0].transpose(0, 1)
        feat_xz = grid_sample(plane_xz, coords[..., [0, 2]],)[0, :, :, 0].transpose(0, 1)
        feat_yz = grid_sample(plane_yz, coords[..., [1, 2]],)[0, :, :, 0].transpose(0, 1)

        # feat = torch.cat([feat_xy, feat_xz, feat_yz], dim=1)
        feat = (feat_xy + feat_xz + feat_yz) / 3
        feat = feat.reshape(*shape[:-1], self.n_output_dims)
        return feat
