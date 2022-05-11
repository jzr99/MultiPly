from .idr_ori_networks import ImplicitNet, RenderingNet
from .ray_tracing import RayTracing
from .sample_network import SampleNetwork
# from .loss import IDRLoss
from ..utils import idr_utils

import numpy as np
import cv2
import os
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl


class IDRNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.implicit_network = ImplicitNet(opt.implicit_network)
        self.rendering_network = RenderingNet(opt.rendering_network)
        self.ray_tracer = RayTracing()
        self.sample_network = SampleNetwork()
        self.object_bounding_sphere = opt.ray_tracer.object_bounding_sphere

    def sdf_func(self, x, **kwargs):
        if hasattr(self, "deformer"):
            x, _ = self.deformer(x, kwargs["cond"], kwargs["tfs"])
        return self.implicit_network(x)[:, 0]

    def forward(self, input):
        # Parse model input
        torch.set_grad_enabled(True)

        uv = input["uv"]
        object_mask = input["object_mask"].reshape(-1)
        ray_dirs, cam_loc = idr_utils.back_project(uv, input["P"], input["C"])
        batch_size, num_pixels, _ = ray_dirs.shape

        self.implicit_network.eval()
        with torch.no_grad():
            points, network_object_mask, dists = self.ray_tracer(
                sdf=lambda x: self.sdf_func(x),
                cam_loc=cam_loc,
                object_mask=object_mask,
                ray_directions=ray_dirs,
                smpl_mesh = None)
        self.implicit_network.train()

        points = (cam_loc.unsqueeze(1) +
                  dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)

        sdf_output = self.implicit_network(points)[:, 0:1]
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.training:
            surface_mask = network_object_mask & object_mask
            surface_points = points[surface_mask]
            surface_dists = dists[surface_mask].unsqueeze(-1)
            surface_ray_dirs = ray_dirs[surface_mask]
            surface_cam_loc = cam_loc.unsqueeze(1).repeat(
                1, num_pixels, 1).reshape(-1, 3)[surface_mask]
            surface_output = sdf_output[surface_mask]
            N = surface_points.shape[0]

            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = batch_size * num_pixels // 2
            eikonal_points = torch.empty(n_eik_points,
                                         3).uniform_(-eik_bounding_box,
                                                     eik_bounding_box).cuda()
            eikonal_pixel_points = points.clone()
            eikonal_pixel_points = eikonal_pixel_points.detach()
            eikonal_points = torch.cat([eikonal_points, eikonal_pixel_points],
                                       0)

            points_all = torch.cat([surface_points, eikonal_points], dim=0)

            output = self.implicit_network(surface_points)
            surface_sdf_values = output[:N, 0:1].detach()

            g = self.implicit_network.gradient(points_all)
            surface_points_grad = g[:N, :].clone().detach()
            grad_theta = g[N:, :]

            differentiable_surface_points = self.sample_network(
                surface_output, surface_sdf_values, surface_points_grad,
                surface_dists, surface_cam_loc, surface_ray_dirs)
        else:
            surface_mask = network_object_mask
            differentiable_surface_points = points[surface_mask]
            grad_theta = None

        view = -ray_dirs[surface_mask]

        rgb_values = torch.ones_like(points).float().cuda()
        if differentiable_surface_points.shape[0] > 0:
            rgb_values[surface_mask] = self.get_rbg_value(
                differentiable_surface_points, view)

        output = {
            'points': points,
            'rgb_values': rgb_values,
            'sdf_output': sdf_output,
            'network_object_mask': network_object_mask,
            'object_mask': object_mask,
            'grad_theta': grad_theta
        }
        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, :]
        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)
        return rgb_vals

class IDRLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.eikonal_weight = 0.1 # opt.eikonal_weight
        self.bone_weight = opt.bone_weight
        self.normal_weight = opt.normal_weight
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
    
    def get_rgb_loss(self, rgb_values, rgb_gt):
        # rgb_gt = rgb_gt.reshape(-1, 3)
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        if rgb_loss == float('nan'):
            import ipdb
            ipdb.set_trace()
        return rgb_loss
    
    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
        return eikonal_loss
        
    def get_normal_loss(self, normal_values, surface_normal_gt, normal_weight):
        # TODO Check
        normal_loss = torch.mean(torch.norm((normal_values-surface_normal_gt) ** 2, dim=1)) # torch.mean(normal_weight[:, None] * torch.norm((normal_values-surface_normal_gt) ** 2, dim=1))
        return normal_loss
    
    def get_bone_loss(self, w_pd, w_gt):
        bone_loss = self.l2_loss(w_pd, w_gt)
        return bone_loss
    
    def forward(self, model_outputs, ground_truth):
        nan_filter = ~torch.any(model_outputs['rgb_values'].isnan(), dim=1)
        rgb_gt = ground_truth['rgb'][0].cuda()
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'][nan_filter], rgb_gt[nan_filter])
        # import ipdb
        # ipdb.set_trace()
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        loss = rgb_loss + self.eikonal_weight * eikonal_loss
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
        }

class IDR(pl.LightningModule):
    def __init__(self, opt) -> None:
        super().__init__()

        # self.automatic_optimization = False
        self.model = IDRNetwork(opt.model)
        self.loss = IDRLoss(opt.model.loss)
        self.training_modules = ["model"]

        self.opt = opt
        for milestone in self.opt.model.alpha_milestones:
            if self.current_epoch > milestone:
                self.loss.opt.alpha *= self.opt.model.alpha_factor

    def configure_optimizers(self):
        params = chain(*[
            getattr(self, module).parameters()
            for module in self.training_modules
        ])

        self.optimizer = optim.Adam(params, lr=self.opt.model.learning_rate)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor)
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def training_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        model_outputs = self.model(inputs)
        loss_output = self.loss(model_outputs, targets)
        for k, v in loss_output.items():
            if k in ["loss"]:
                self.log(k, v.item())
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)
        return loss_output["loss"]

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch in self.opt.model.alpha_milestones:
            self.loss.opt.alpha *= self.opt.model.alpha_factor
        return super().training_epoch_end(outputs)

    def validation_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        model_outputs = self.model(inputs)
        return {
            "rgb_pred": model_outputs["rgb_values"].detach().clone(),
            **targets,
        }

    def validation_epoch_end(self, outputs) -> None:
        img_size = outputs[0]["img_size"].squeeze(0)

        rgb_pred = torch.cat([output["rgb_pred"] for output in outputs], dim=0)
        rgb_pred = (rgb_pred.reshape(*img_size, -1) + 1) / 2

        rgb_gt = torch.cat([output["rgb"] for output in outputs], dim=1).squeeze(0)
        rgb_gt = (rgb_gt.reshape(*img_size, -1) + 1) / 2

        rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)

        os.makedirs("rendering", exist_ok=True)
        cv2.imwrite(f"rendering/{self.current_epoch:04d}.png", rgb[:, :, ::-1])