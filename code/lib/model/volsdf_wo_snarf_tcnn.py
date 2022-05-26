from .volsdf_ori_tcnn_networks import ImplicitNet, RenderingNet
from .ray_tracing import RayTracing
from .sample_network import SampleNetwork
from .density import LaplaceDensity
from .ray_sampler import ErrorBoundSampler
from .deformer import ForwardDeformer
from .smpl import SMPLServer
from .sampler import PointInSpace, PointOnBones
from .loss import VolSDFLoss
from ..utils import idr_utils
from ..utils.mesh import generate_mesh
from ..utils.snarf_utils import weights2colors
from ..utils import rend_util

import numpy as np
import cv2
import trimesh
import os
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import pytorch_lightning as pl
import hydra

class VolSDFNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.implicit_network = ImplicitNet(opt.implicit_network)
        self.rendering_network = RenderingNet(opt.rendering_network)
        # self.ray_tracer = RayTracing(**opt.ray_tracer)
        self.sampler = PointInSpace()
        # self.object_bounding_sphere = opt.ray_tracer.object_bounding_sphere
        # self.deformer = ForwardDeformer(opt.deformer, betas=betas)
        self.scene_bounding_sphere = opt.implicit_network.scene_bounding_sphere

        self.sphere_scale = opt.implicit_network.sphere_scale
        self.white_bkgd = opt.white_bkgd
        self.bg_color = torch.tensor([1.0, 1.0, 1.0]).float().cuda()

        self.density = LaplaceDensity(**opt.density)
        self.ray_sampler = ErrorBoundSampler(self.scene_bounding_sphere, **opt.ray_sampler)
        self.use_body_pasing = opt.use_body_parsing
        if opt.smpl_init:
            smpl_model_state = torch.load(hydra.utils.to_absolute_path('./outputs/smpl_init_%s_512.pth' % gender))
            self.implicit_network.load_state_dict(smpl_model_state["model_state_dict"])
            # self.deformer.load_state_dict(smpl_model_state["deformer_state_dict"])

    def forward(self, input):
        # Parse model input
        torch.set_grad_enabled(True)

        intrinsics = input["intrinsics"]
        pose = input["pose"]
        uv = input["uv"]

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)

        # ray_dirs, cam_loc = idr_utils.back_project(uv, input["P"], input["C"])

        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self)
        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)

        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)

        sdf, feature_vectors, gradients = self.implicit_network.get_outputs(points_flat)

        rgb_flat = self.rendering_network(points_flat, gradients, dirs_flat, feature_vectors)
        rgb = rgb_flat.reshape(-1, N_samples, 3)

        weights = self.volume_rendering(z_vals, sdf)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        if self.training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_pixels
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-self.scene_bounding_sphere, self.scene_bounding_sphere).cuda()

            # add some of the near surface points
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(-1, 3)
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 0)
            grad_theta = self.implicit_network.gradient(eikonal_points)
        else:
            gradients = gradients.detach()
            normals = gradients / gradients.norm(2, -1, keepdim=True)
            normals = normals.reshape(-1, N_samples, 3)
            normal_values = torch.sum(weights.unsqueeze(-1) * normals, 1)


        view = -dirs.reshape(-1, 3) # view = -ray_dirs[surface_mask]


        if self.training:
            output = {
                'rgb_values': rgb_values,
                'grad_theta': grad_theta
            }
        else:

            output = {
                'rgb_values': rgb_values,
                'normal_values': normal_values,
            }
        return output


    def volume_rendering(self, z_vals, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1])  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(dists.shape[0], 1)], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy[:, :-1]], dim=-1)  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        weights = alpha * transmittance # probability of the ray hits something here

        return weights

def gradient(inputs, outputs):

    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, :, -3:]
    return points_grad


class TCNNVolSDFLoss(nn.Module):
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

class VolSDF(pl.LightningModule):
    def __init__(self, opt) -> None:
        super().__init__()
        self.model = VolSDFNetwork(opt.model)
        self.loss = TCNNVolSDFLoss(opt.model.loss)
        self.training_modules = ["model"]
        self.opt = opt

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

    def validation_step(self, batch, *args, **kwargs):
        inputs, targets = batch
        model_outputs = self.model(inputs)
        return {
            "rgb_values": model_outputs["rgb_values"].detach().clone(),
            "normal_values": model_outputs["normal_values"].detach().clone(),
            **targets,
        }

    def validation_epoch_end(self, outputs) -> None:
        img_size = outputs[0]["img_size"].squeeze(0)

        rgb_pred = torch.cat([output["rgb_values"] for output in outputs], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([output["normal_values"] for output in outputs], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        rgb_gt = torch.cat([output["rgb"] for output in outputs], dim=1).squeeze(0)
        rgb_gt = rgb_gt.reshape(*img_size, -1)

        if 'normal' in outputs[0].keys():
            normal_gt = torch.cat([output["normal"] for output in outputs], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        os.makedirs("rendering", exist_ok=True)
        os.makedirs("normal", exist_ok=True)

        cv2.imwrite(f"rendering/{self.current_epoch}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch}.png", normal[:, :, ::-1])

    def test_step(self, batch, *args, **kwargs):
        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_batches = (total_pixels + pixel_per_batch -
                       1) // pixel_per_batch
        results = []
        os.makedirs("test_rendering", exist_ok=True)
        for i in range(num_batches):
            print("current batch: %d / %d" % (i, num_batches))
            indices = list(range(i * pixel_per_batch,
                                min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"uv": inputs["uv"][:, indices],
                            "intrinsics": inputs["intrinsics"],
                            "pose": inputs["pose"]}
            if self.opt.model.use_body_parsing:
                batch_inputs['body_parsing'] = inputs['body_parsing'][:, indices]
            
            if 'rgb' in targets.keys():
                batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone(),
                                "img_size": targets["img_size"]}
            else:
                batch_targets = {"img_size": targets["img_size"]}
            # torch.cuda.memory_stats(device="cuda:0")
            torch.cuda.empty_cache()
            with torch.no_grad():
                model_outputs = self.model(batch_inputs)
            results.append({'rgb_values':model_outputs["rgb_values"].detach().clone(), 
                            'normal_values': model_outputs["normal_values"].detach().clone(),
                            **batch_targets})         
        img_size = results[0]["img_size"].squeeze(0)
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([result["normal_values"] for result in results], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        if 'rgb' in results[0].keys():
            rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            rgb_gt = rgb_gt.reshape(*img_size, -1)
            rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        else:
            rgb = torch.cat([rgb_pred], dim=0).cpu().numpy()

        if 'normal' in results[0].keys():
            normal_gt = torch.cat([result["normal"] for result in results], dim=1).squeeze(0)
            normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
        else:
            normal = torch.cat([normal_pred], dim=0).cpu().numpy()

        
        rgb = (rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        os.makedirs("test_rendering", exist_ok=True)
        os.makedirs("test_normal", exist_ok=True)

        cv2.imwrite(f"test_rendering/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"test_normal/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])