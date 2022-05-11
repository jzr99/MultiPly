from .networks import ImplicitNet, RenderingNet
from .ray_tracing import RayTracing
from .sample_network import SampleNetwork
from .density import LaplaceDensity
from .ray_sampler import ErrorBoundSampler, BBoxSampler
from .deformer import ForwardDeformer, SMPLDeformer
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
import open3d as o3d
import pytorch3d

class VolSDFNetwork(nn.Module):
    def __init__(self, opt, betas_path):
        super().__init__()
        self.implicit_network = ImplicitNet(opt.implicit_network)
        self.rendering_network = RenderingNet(opt.rendering_network)
        # self.ray_tracer = RayTracing(**opt.ray_tracer)
        self.sampler = PointInSpace()
        # self.object_bounding_sphere = opt.ray_tracer.object_bounding_sphere
        betas = np.load(betas_path)
        self.use_smpl_deformer = opt.use_smpl_deformer
        if self.use_smpl_deformer:
            self.deformer = SMPLDeformer(betas=betas) 
        else:
            self.deformer = ForwardDeformer(opt.deformer, betas=betas)
        gender = 'male'
        self.sdf_bounding_sphere = opt.implicit_network.scene_bounding_sphere
        self.sphere_scale = opt.implicit_network.sphere_scale
        self.white_bkgd = opt.white_bkgd
        self.bg_color = torch.tensor([1.0, 1.0, 1.0]).float().cuda()

        self.density = LaplaceDensity(**opt.density)
        
        self.use_bbox_sampler = opt.use_bbox_sampler
        if self.use_bbox_sampler:
            self.ray_sampler = BBoxSampler(**opt.ray_sampler)
        else:
            self.ray_sampler = ErrorBoundSampler(self.sdf_bounding_sphere, **opt.ray_sampler)
        self.smpl_server = SMPLServer(gender=gender) # average shape for now. Adjust gender later!
        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)
        self.use_body_pasing = opt.use_body_parsing
        if opt.smpl_init:
            smpl_model_state = torch.load(hydra.utils.to_absolute_path('./outputs/smpl_init_%s_256.pth' % gender))
            self.implicit_network.load_state_dict(smpl_model_state["model_state_dict"])
            if not self.use_smpl_deformer:
                self.deformer.load_state_dict(smpl_model_state["deformer_state_dict"])

    def extract_normal(self, x_c, cond, tfs):
        
        x_c = x_c.unsqueeze(0)
        x_c.requires_grad_(True) 
        output = self.implicit_network(x_c, cond)[..., 0:1]
        gradient_c = gradient(x_c, output)
        gradient_d = self.deformer.forward_skinning_normal(x_c, gradient_c, cond, tfs)

        return gradient_d
    def sdf_func(self, x, cond, smpl_tfs, eval_mode=False):
        if hasattr(self, "deformer"):
            x_c, others = self.deformer(x, None, smpl_tfs, eval_mode)
            x_c = x_c.squeeze(0)
            num_point, num_init, num_dim = x_c.shape
            x_c = x_c.reshape(num_point * num_init, num_dim)
            output = self.implicit_network(x_c, cond)[0].reshape(num_point, num_init, -1)
            sdf = output[:, :, 0]

            feature = output[:, :, 1:]
            if others['valid_ids'].ndim == 3:
                sdf_mask = others['valid_ids'].squeeze(0)
            elif others['valid_ids'].ndim == 2:
                sdf_mask = others['valid_ids']
            sdf[~sdf_mask] = 4.
            sdf, index = torch.min(sdf, dim=1)

            ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
            if self.sdf_bounding_sphere > 0.0:
                sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
                sdf = torch.minimum(sdf.unsqueeze(-1), sphere_sdf)

            x_c = x_c.reshape(num_point, num_init, num_dim)
            x_c = torch.gather(x_c, dim=1, index=index.unsqueeze(-1).unsqueeze(-1).expand(num_point, num_init, num_dim))[:, 0, :]
            feature = torch.gather(feature, dim=1, index=index.unsqueeze(-1).unsqueeze(-1).expand(num_point, num_init, feature.shape[-1]))[:, 0, :]
        return sdf, x_c, feature # [:, 0]

    def sdf_func_with_smpl_deformer(self, x, cond, smpl_tfs, smpl_verts):
        if hasattr(self, "deformer"):
            x_c = self.deformer.forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
            output = self.implicit_network(x_c, cond)[0]
            sdf = output[:, 0:1]
            ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
            if self.sdf_bounding_sphere > 0.0:
                sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
                sdf = torch.minimum(sdf, sphere_sdf)
            feature = output[:, 1:]

        return sdf, x_c, feature
    def forward(self, input):
        # Parse model input
        torch.set_grad_enabled(True)
        # for param in self.deformer.parameters():
        #     param.requires_grad = False
        intrinsics = input["intrinsics"]
        pose = input["pose"]
        uv = input["uv"]

        object_mask = input["object_mask"].reshape(-1)
        if self.use_body_pasing:
            body_parsing = input['body_parsing'].reshape(-1)

        smpl_params = input['smpl_params']
        # smpl_thetas = smpl_params[:, 4:76]
        smpl_pose = input["smpl_pose"]
        # scale, transl, _, betas = torch.split(smpl_params, [1, 3, 72, 10], dim=1)
        scale = smpl_params[:, 0]
        smpl_shape = input["smpl_shape"]
        smpl_trans = input["smpl_trans"]
        smpl_output = self.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
        # smpl_params = smpl_params.detach()
        smpl_tfs = smpl_output['smpl_tfs']

        smpl_mesh = trimesh.Trimesh(vertices=smpl_output['smpl_verts'][0].detach().cpu().numpy(), faces=self.smpl_server.faces, process=False)
        cond = {'smpl': smpl_pose[:, 3:]/np.pi}
        # ray_dirs, cam_loc = idr_utils.back_project(uv, input["P"], input["C"])
        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape
        """
        # self.implicit_network.eval()
        # with torch.no_grad():
        #     points, network_object_mask, dists = self.ray_tracer(
        #         sdf=lambda x: self.sdf_func(x, cond, smpl_tfs, eval_mode=True)[0],
        #         cam_loc=cam_loc,
        #         object_mask=object_mask,
        #         ray_directions=ray_dirs,
        #         smpl_mesh = smpl_mesh)
        # self.implicit_network.train()

        # points = (cam_loc.unsqueeze(1) +
        #           dists.reshape(batch_size, num_pixels, 1) * ray_dirs).reshape(-1, 3)
        """
        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)

        if self.use_smpl_deformer:
            if self.use_bbox_sampler:
                z_vals = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, smpl_mesh, self.training)
            else:    
                z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, cond, smpl_tfs, eval_mode=True, smpl_verts=smpl_output['smpl_verts'])
        else:
            if self.use_bbox_sampler:
                z_vals = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, smpl_mesh, self.training)
            else:
                z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, cond, smpl_tfs, eval_mode=True)

        N_samples = z_vals.shape[1]

        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        points_flat = points.reshape(-1, 3)


        dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
        dirs_flat = dirs.reshape(-1, 3)
        if self.use_smpl_deformer:
            sdf_output, canonical_points, _ = self.sdf_func_with_smpl_deformer(points_flat, cond, smpl_tfs, smpl_output['smpl_verts'])
        else:
            sdf_output, canonical_points, _ = self.sdf_func(points_flat, cond, smpl_tfs, eval_mode=True)
        sdf_output = sdf_output.unsqueeze(1)

        # if self.use_smpl_deformer:
        #   _, canonical_smpl_points, _ = self.sdf_func_with_smpl_deformer(smpl_output['smpl_verts'][0], cond, smpl_tfs, smpl_output['smpl_verts']) 
        # else:
        #   _, canonical_smpl_points, _ = self.sdf_func(smpl_output['smpl_verts'][0], cond, smpl_tfs, eval_mode=True)
        # posed_smpl_points = self.deformer.forward_skinning(canonical_smpl_points[None], cond, smpl_tfs)
        # import ipdb
        # ipdb.set_trace()
        # o3d_pcl = o3d.geometry.PointCloud()
        # o3d_pcl.points = o3d.utility.Vector3dVector(points_flat.detach().cpu().numpy().squeeze())
        # o3d.io.write_point_cloud("/home/chen/Desktop/points_flat.ply", o3d_pcl)
        # o3d_cano_pcl = o3d.geometry.PointCloud()
        # o3d_cano_pcl.points = o3d.utility.Vector3dVector(canonical_points.detach().cpu().numpy().squeeze())
        # o3d.io.write_point_cloud("/home/chen/Desktop/canonical_points.ply", o3d_cano_pcl)
        # smpl_pcl = o3d.geometry.PointCloud()
        # smpl_pcl.points = o3d.utility.Vector3dVector(smpl_output['smpl_verts'][0].detach().cpu().numpy().squeeze())
        # o3d.io.write_point_cloud("/home/chen/Desktop/smpl_pcl.ply", smpl_pcl)
        # o3d_cano_smpl_pcl = o3d.geometry.PointCloud()
        # o3d_cano_smpl_pcl.points = o3d.utility.Vector3dVector(canonical_smpl_points.detach().cpu().numpy().squeeze())
        # o3d.io.write_point_cloud("/home/chen/Desktop/canonical_smpl_pcl.ply", o3d_cano_smpl_pcl)
        # o3d_posed_smpl_pcl = o3d.geometry.PointCloud()
        # o3d_posed_smpl_pcl.points = o3d.utility.Vector3dVector(posed_smpl_points.detach().cpu().numpy().squeeze())
        # o3d.io.write_point_cloud("/home/chen/Desktop/posed_smpl_pcl.ply", o3d_posed_smpl_pcl)
        # ipdb.set_trace()

        if self.training:
            canonical_points = canonical_points.reshape(num_pixels, N_samples, 3) # [surface_mask]

            cam_loc = cam_loc.unsqueeze(1).repeat(1, N_samples, 1) # [surface_mask]

            # normal = input["normal"].reshape(-1, 3)
            # surface_normal = normal # [surface_mask]

            # N = surface_points.shape[0]

            canonical_points = canonical_points.reshape(-1, 3)

            cam_loc = cam_loc.reshape(-1, 3)

            # sample canonical SMPL surface pnts for the eikonal loss
            smpl_verts_c = self.smpl_server.verts_c.repeat(batch_size, 1,1)
            
            indices = torch.randperm(smpl_verts_c.shape[1])[:num_pixels].cuda()
            verts_c = torch.index_select(smpl_verts_c, 1, indices)
            sample = self.sampler.get_points(verts_c, global_ratio=0.)
            # sample = torch.cat([sample_local, sample_global], dim=1)
            sample.requires_grad_()
            local_pred = self.implicit_network(sample, cond)[..., 0:1]
            grad_theta = gradient(sample, local_pred)

            differentiable_points = canonical_points 

            # pts_c_sdf, sdf_gt = self.sampler_bone.get_points(self.smpl_server.joints_c.expand(1, -1, -1))
            # sdf_pd = self.implicit_network(pts_c_sdf[0], cond)[..., 0]

            # Bone regularization!!!
            if self.use_smpl_deformer:
                w_gt, w_pd = None, None
            else:
                pts_c_w, w_gt = self.sampler_bone.get_joints(self.smpl_server.joints_c.expand(1, -1, -1))
                w_pd = self.deformer.query_weights(pts_c_w, cond)

            # SMPL skinning weight regularization!!!
            # lbs_weight = self.deformer.query_weights(surface_canonical_points.detach().unsqueeze(0), cond)[0]
            # gt_lbs_weight = self.deformer.query_smpl_weights(surface_canonical_points.detach().unsqueeze(0))[0]
            # skinning_values = torch.ones([points.shape[0], points.shape[1], lbs_weight.shape[1]]).float().cuda()
            # gt_skinning_values = torch.ones([points.shape[0], points.shape[1], lbs_weight.shape[1]]).float().cuda()
            # skinning_values = skinning_values[surface_mask].reshape(-1, lbs_weight.shape[1])
            # gt_skinning_values = gt_skinning_values[surface_mask].reshape(-1, lbs_weight.shape[1])
            # skinning_values = lbs_weight
            # gt_skinning_values = gt_lbs_weight
            normal_weight = torch.zeros(points_flat.shape[0]).float().cuda()
        else:
            differentiable_points = canonical_points.reshape(num_pixels, N_samples, 3).reshape(-1, 3)
            grad_theta = None

        sdf_output = sdf_output.reshape(num_pixels, N_samples, 1).reshape(-1, 1)
        z_vals = z_vals # [surface_mask]
        view = -dirs.reshape(-1, 3) # view = -ray_dirs[surface_mask]

        if differentiable_points.shape[0] > 0:
            rgb_values, others = self.get_rbg_value(differentiable_points, view,
                                                    cond, smpl_tfs, is_training=self.training)                       
            normal_values = others['normals'] # should we flip the normals? No

        rgb_values = rgb_values.reshape(-1, N_samples, 3)
        normal_values = normal_values.reshape(-1, N_samples, 3)
        weights = self.volume_rendering(z_vals, sdf_output)

        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb_values, 1)
        # rgb_values = weights.sum(-1, keepdims=True)[..., [0] * 3]
        normal_values = torch.sum(weights.unsqueeze(-1) * normal_values, 1)

        # white background assumption
        if self.white_bkgd:
            acc_map = torch.sum(weights, -1)
            rgb_values = rgb_values + (1. - acc_map[..., None]) * self.bg_color.unsqueeze(0)

        if self.training:
            if differentiable_points.shape[0] > 0:
                normal_weight = torch.einsum('ij, ij->i', normal_values, -ray_dirs).detach()
            output = {
                'points': points,
                'rgb_values': rgb_values,
                'normal_values': normal_values,
                # 'surface_normal_gt': surface_normal,
                'normal_weight': normal_weight,
                'sdf_output': sdf_output,
                'object_mask': object_mask,
                # "sdf_pd": sdf_pd,
                # "sdf_gt": sdf_gt,
                "w_pd": w_pd,
                "w_gt": w_gt,
                # 'lbs_weight_pd': skinning_values,
                # 'lbs_weight_gt': gt_skinning_values,
                'grad_theta': grad_theta,
                'use_smpl_deformer': self.use_smpl_deformer
            }
        else:
            output = {
                'points': points,
                'rgb_values': rgb_values,
                'normal_values': normal_values,
                'sdf_output': sdf_output,
                # 'network_object_mask': network_object_mask,
                # 'object_mask': object_mask
            }
        return output


    def get_rbg_value(self, points, view_dirs, cond, tfs, surface_body_parsing=None, is_training=True):
        pnts_c = points
        others = {}

        _, gradients, feature_vectors = self.forward_gradient(pnts_c, cond, tfs, create_graph=is_training, retain_graph=is_training)
        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6) # nn.functional.normalize(gradients, dim=-1, eps=1e-6) gradients
        rgb_vals = self.rendering_network(pnts_c, normals.detach(), view_dirs, cond['smpl'],
                                          feature_vectors, surface_body_parsing)
        
        others['normals'] = normals
        return rgb_vals, others

    def forward_gradient(self, pnts_c, cond, tfs, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)

        pnts_d = self.deformer.forward_skinning(pnts_c.unsqueeze(0), None, tfs).squeeze(0)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            grad = torch.autograd.grad(
                outputs=pnts_d,
                inputs=pnts_c,
                grad_outputs=d_out,
                create_graph=create_graph,
                retain_graph=True if i < num_dim - 1 else retain_graph,
                only_inputs=True)[0]
            grads.append(grad)
        grads = torch.stack(grads, dim=-2)
        grads_inv = grads.inverse()

        output = self.implicit_network(pnts_c, cond)[0]
        sdf = output[:, :1]

        if self.sdf_bounding_sphere > 0.0:
            sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - pnts_c.norm(2,1, keepdim=True))
            sdf = torch.minimum(sdf, sphere_sdf)
        
        feature = output[:, 1:]
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        gradients = torch.autograd.grad(
            outputs=sdf,
            inputs=pnts_c,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]

        return grads.reshape(grads.shape[0], -1), torch.nn.functional.normalize(torch.einsum('bi,bij->bj', gradients, grads_inv), dim=1), feature

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

class VolSDF(pl.LightningModule):
    def __init__(self, opt, betas_path) -> None:
        super().__init__()
        # self.automatic_optimization = False
        self.model = VolSDFNetwork(opt.model, betas_path)
        self.loss = VolSDFLoss(opt.model.loss)
        self.training_modules = ["model"]
        # self.smpl_pose = nn.Parameter(torch.zeros(1, 72))
        # self.smpl_shape = nn.Parameter(torch.zeros(1, 10))
        # self.smpl_trans = nn.Parameter(torch.zeros(1, 3))
        self.smpl_pose = torch.tensor(np.zeros([1, 72]), requires_grad=True, dtype=torch.float32)
        self.smpl_shape = torch.tensor(np.zeros([1, 10]), requires_grad=True, dtype=torch.float32)
        self.smpl_trans = torch.tensor(np.zeros([1, 3]), requires_grad=True, dtype=torch.float32)
        self.opt = opt

    def configure_optimizers(self):
        params = chain(*[
            getattr(self, module).parameters()
            for module in self.training_modules
        ])

        self.optimizer = optim.Adam(params, lr=self.opt.model.learning_rate)
        # self.pose_optimizer = optim.Adam([self.smpl_pose , self.smpl_shape, self.smpl_trans], lr=1e-4) 
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor)
        # self._optimizers = [self.optimizer]
        return [{"optimizer": self.optimizer, "lr_scheduler": self.scheduler}]

    def training_step(self, batch):
        inputs, targets = batch

        batch_idx = inputs["idx"]
        
        device = inputs["smpl_params"].device
        if self.current_epoch < 10000:
            reference_pose = inputs["smpl_params"][:, 4:76]
            reference_shape = inputs["smpl_params"][:, 76:]
            reference_trans = inputs["smpl_params"][:, 1:4]
            self.smpl_pose.data = reference_pose
            self.smpl_shape.data = reference_shape
            self.smpl_trans.data = reference_trans
        else:
            smpl_params = np.load(f"./opt_params/smpl_params_{batch_idx[0]:04d}.npz")
            self.smpl_pose.data = torch.from_numpy(smpl_params["smpl_pose"]).to(device)
            self.smpl_shape.data = torch.from_numpy(smpl_params["smpl_shape"]).to(device)
            self.smpl_trans.data = torch.from_numpy(smpl_params["smpl_trans"]).to(device)

        inputs["smpl_pose"] = self.smpl_pose
        inputs["smpl_shape"] = self.smpl_shape
        inputs["smpl_trans"] = self.smpl_trans

        model_outputs = self.model(inputs)
        # if model_outputs is None:
        #     for optimizer in self._optimizers:
        #         optimizer.zero_grad()
        #     pass
        # else:

        loss_output = self.loss(model_outputs, targets)
        for k, v in loss_output.items():
            if k in ["loss"]:
                self.log(k, v.item(), prog_bar=True, on_step=True)
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)

            # if self.current_epoch < 10000:
                # self.model.implicit_network.eval()
                # self.model.deformer.eval()
                # for param in self.model.implicit_network.parameters():
                #     param.requires_grad = False
                # for param in self.model.deformer.parameters():
                    # param.requires_grad = False
            # else:
                # self.model.implicit_network.train()
                # self.model.deformer.train()
                # for param in self.model.deformer.parameters():
                #     param.requires_grad = False

            # for optimizer in self._optimizers:
            #     optimizer.zero_grad()

            # loss_output["loss"].backward()
            # self.manual_backward(loss_output["loss"])
            # for optimizer in self._optimizers:
            #     optimizer.step()

            # if not os.path.exists("./opt_params"):
            #     os.makedirs("./opt_params")
            # smpl_params = {"smpl_pose":self.smpl_pose.detach().cpu().numpy(), 
            #             "smpl_shape":self.smpl_shape.detach().cpu().numpy(),
            #             "smpl_trans":self.smpl_trans.detach().cpu().numpy()}

            # np.savez(f"./opt_params/smpl_params_{batch_idx[0]:04d}.npz", **smpl_params)
        return loss_output["loss"]

    def training_epoch_end(self, outputs) -> None:
        if self.current_epoch in self.opt.model.alpha_milestones:
            self.loss.alpha *= self.opt.model.alpha_factor
        # if self.current_epoch % 1 == 0:
        #     self.save_checkpoints(self.current_epoch)
        return super().training_epoch_end(outputs)

    def query_oc(self, x, cond):
        
        x = x.reshape(-1, 3)
        mnfld_pred = self.model.implicit_network(x, cond)[:,:,0].reshape(-1,1)
        # mnfld_pred = self.model.density(mnfld_pred)
        return {'occ':mnfld_pred}

    def query_wc(self, x, cond):
        
        x = x.reshape(-1, 3)
        w = self.model.deformer.query_weights(x, cond)
    
        return w

    def query_od(self, x, cond, smpl_tfs):
        
        x = x.reshape(-1, 3)
        x_c, others = self.model.deformer(x, cond, smpl_tfs, eval_mode = True)
        x_c = x_c.squeeze(0)
        num_point, num_init, num_dim = x_c.shape

        x_c = x_c.reshape(num_point * num_init, num_dim)
        output = self.model.implicit_network(x_c, cond)[0].reshape(num_point, num_init, -1)
        sdf = output[:, :, 0]
        if others['valid_ids'].ndim == 3:
            sdf_mask = others['valid_ids'].squeeze(0)
        elif others['valid_ids'].ndim == 2:
            sdf_mask = others['valid_ids']
        sdf[~sdf_mask] = 1.

        sdf, _ = torch.min(sdf, dim=1)
        sdf = sdf.reshape(-1,1)
        
        return {'occ': sdf}
    
    def validation_step(self, batch, *args, **kwargs):

        output = {}
        inputs, targets = batch

        self.model.eval()

        device = inputs["smpl_params"].device
        if self.current_epoch < 10000:
            reference_pose = inputs["smpl_params"][:, 4:76]
            reference_shape = inputs["smpl_params"][:, 76:]
            reference_trans = inputs["smpl_params"][:, 1:4]
            val_smpl_pose = reference_pose
            val_smpl_shape = reference_shape
            val_smpl_trans = reference_trans
        else:
            val_smpl_params = np.load(f"./opt_params/smpl_params_0000.npz")
            val_smpl_pose = torch.from_numpy(val_smpl_params["smpl_pose"]).to(device)
            val_smpl_shape = torch.from_numpy(val_smpl_params["smpl_shape"]).to(device)
            val_smpl_trans = torch.from_numpy(val_smpl_params["smpl_trans"]).to(device)
        inputs["smpl_pose"] = val_smpl_pose
        inputs["smpl_shape"] = val_smpl_shape
        inputs["smpl_trans"] = val_smpl_trans

        cond = {'smpl': inputs["smpl_pose"][:, 3:]/np.pi}
        mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=3)
        verts_mesh  = torch.tensor(mesh_canonical.vertices).cuda().float().unsqueeze(0)
        weights_mesh = self.query_wc(verts_mesh, cond).data.cpu().numpy()

        mesh_canonical.colors = weights2colors(weights_mesh)*255
        
        mesh_canonical = trimesh.Trimesh(mesh_canonical.vertices, mesh_canonical.faces, vertex_colors=mesh_canonical.colors)
        
        # model_outputs = self.model(inputs)

        output.update({
            'canonical_weighted':mesh_canonical
        })

        split = idr_utils.split_input(inputs, targets["total_pixels"][0], n_pixels=min(targets['pixel_per_batch'], targets["img_size"][0,0] * targets["img_size"][0,1]))

        res = []
        for s in split:

            out = self.model(s)

            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except:
                    out[k] = v

            res.append({
                'points': out['points'].detach(),
                'rgb_values': out['rgb_values'].detach(),
                'normal_values': out['normal_values'].detach(),
                # 'network_object_mask': out['network_object_mask'].detach(),
                # 'object_mask': out['object_mask'].detach()
            })
        batch_size = targets['rgb'].shape[0]

        model_outputs = idr_utils.merge_output(res, targets["total_pixels"][0], batch_size)

        output.update({
            "rgb_values": model_outputs["rgb_values"].detach().clone(),
            "normal_values": model_outputs["normal_values"].detach().clone(),
            **targets,
        })
            
        # return {
        #     "rgb_values": model_outputs["rgb_values"].detach().clone(),
        #     "normal_values": model_outputs["normal_values"].detach().clone(),
        #     **targets,
        # }

        return output

    def validation_step_end(self, batch_parts):
        return batch_parts

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

        canonical_mesh = outputs[0]['canonical_weighted']
        canonical_mesh.export(f"rendering/{self.current_epoch}.ply")

        cv2.imwrite(f"rendering/{self.current_epoch}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch}.png", normal[:, :, ::-1])
    
    def test_step(self, batch, *args, **kwargs):
        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_batches = (total_pixels + pixel_per_batch -
                       1) // pixel_per_batch
        results = []
        os.makedirs("test_rendering", exist_ok=True)
        for i in range(num_batches):
            print("current batch:", i)
            indices = list(range(i * pixel_per_batch,
                                min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"object_mask": inputs["object_mask"][:, indices],
                            "uv": inputs["uv"][:, indices],
                            "P": inputs["P"],
                            "C": inputs["C"],
                            "smpl_params": inputs["smpl_params"]}
            if self.opt.model.use_body_parsing:
                batch_inputs['body_parsing'] = inputs['body_parsing'][:, indices]
            batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone(),
                             "img_size": targets["img_size"]}

            # torch.cuda.memory_stats(device="cuda:0")
            torch.cuda.empty_cache()
            with torch.no_grad():
                model_outputs = self.model(batch_inputs)
            results.append({'rgb_values':model_outputs["rgb_values"].detach().clone(), 
                            **batch_targets})         
        img_size = results[0]["img_size"].squeeze(0)
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = (rgb_pred.reshape(*img_size, -1) + 1) / 2

        rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
        rgb_gt = (rgb_gt.reshape(*img_size, -1) + 1) / 2

        rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)

        cv2.imwrite(f"rendering/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
