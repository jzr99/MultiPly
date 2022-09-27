from .networks import ImplicitNet, RenderingNet
from .ray_tracing import RayTracing
from .sample_network import SampleNetwork
from .fast_snarf import ForwardDeformer
from .smpl import SMPLServer
from .sampler import PointInSpace, PointOnBones
from .loss import ThreeDLoss

import numpy as np
import cv2
import os
from itertools import chain
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import pytorch_lightning as pl
import kaolin
from kaolin.ops.mesh import index_vertices_by_faces

class SDF_Init_Network(nn.Module):
    def __init__(self, opt):
        super().__init__()

        self.implicit_network = ImplicitNet(opt.implicit_network)
        betas = np.zeros(10)
        self.smpl_server = SMPLServer(gender='male', betas=betas) # average shape for now. Adjust gender later!
        self.deformer = ForwardDeformer(opt.deformer, self.smpl_server)
        self.sampler = PointInSpace()
        self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)
        self.points_batch = 2048
        self.smpl_faces = torch.tensor(self.smpl_server.smpl.faces.astype('int')).cuda().unsqueeze(0)
    # def sdf_func(self, x, cond, smpl_tfs, eval_mode=False):
    #     if hasattr(self, "deformer"):
    #         x_c, others = self.deformer(x, None, smpl_tfs, eval_mode)
    #         x_c = x_c.squeeze(0)
    #         num_point, num_init, num_dim = x_c.shape
    #         x_c = x_c.reshape(num_point * num_init, num_dim)
    #         output = self.implicit_network(x_c, cond)[0].reshape(num_point, num_init, -1)
    #         sdf = output[:, :, 0]
    #         feature = output[:, :, 1:]
    #         if others['valid_ids'].ndim == 3:
    #             sdf_mask = others['valid_ids'].squeeze(0)
    #         elif others['valid_ids'].ndim == 2:
    #             sdf_mask = others['valid_ids']
    #         sdf[~sdf_mask] = 1.
    #         sdf, index = torch.min(sdf, dim=1)

    #         x_c = x_c.reshape(num_point, num_init, num_dim)
    #         x_c = torch.gather(x_c, dim=1, index=index.unsqueeze(-1).unsqueeze(-1).expand(num_point, num_init, num_dim))[:, 0, :]
    #         feature = torch.gather(feature, dim=1, index=index.unsqueeze(-1).unsqueeze(-1).expand(num_point, num_init, feature.shape[-1]))[:, 0, :]
    #     return sdf, x_c, feature # [:, 0]

    def forward(self, input):
        # Parse model input
        torch.set_grad_enabled(True)
        self.implicit_network.train()

        smpl_params = input['smpl_params']
        smpl_thetas = smpl_params[:, 4:76]
        scale = smpl_params[:, 0:1]
        smpl_pose = input["smpl_params"][:, 4:76]
        smpl_shape = input["smpl_params"][:, 76:]
        smpl_trans = input["smpl_params"][:, 1:4]
        smpl_output = self.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
        smpl_tfs = smpl_output['smpl_tfs']
        smpl_verts = smpl_output['smpl_verts']
        # from IPython import embed; embed()

        num_batch, num_verts, num_dim = smpl_verts.shape
        random_idx = []
        for _ in range(num_batch):
            r = torch.randperm(num_verts).cuda()[:self.points_batch].unsqueeze(0).unsqueeze(-1)
            random_idx.append(r)
        random_idx = torch.cat(random_idx, 0).expand(-1, -1, num_dim)
        mnfld_pnts = torch.gather(smpl_verts, 1, random_idx)
        mnfld_pnts = self.sampler.get_points(mnfld_pnts)

        gt_sign = kaolin.ops.mesh.check_sign(smpl_verts, self.smpl_faces[0], mnfld_pnts).float().unsqueeze(-1)
        gt_sign = 1 - 2*gt_sign
        # gt_df = torch.sqrt(kaolin.metrics.trianglemesh.point_to_mesh_distance(mnfld_pnts, smpl_verts, self.smpl_faces[0])[0]) 
        smpl_face_vertices = index_vertices_by_faces(smpl_verts, self.smpl_faces[0])
        gt_df = torch.sqrt(kaolin.metrics.trianglemesh.point_to_mesh_distance(mnfld_pnts, smpl_face_vertices)[0]) 
        mnfld_pnts.requires_grad_()
        gt_sdf = gt_sign[...,0] * gt_df
        
        cond = {'smpl': smpl_thetas[:,3:]/np.pi}

        mnfld_pnts_c, others = self.deformer(mnfld_pnts, cond, smpl_tfs)

        if self.training:

            num_batch, num_point, num_init, num_dim = mnfld_pnts_c.shape
            mnfld_pnts_c = mnfld_pnts_c.reshape(num_batch, num_point * num_init, num_dim)
            valid_ids = others['valid_ids'].reshape(num_batch, num_point * num_init, 1)
            valid_scaler = torch.zeros_like(valid_ids).float()
            valid_scaler[valid_ids] = 1
            mnfld_pnts_c.requires_grad_()

            mnfld_pred = self.implicit_network(mnfld_pnts_c, cond)[..., 0:1]
            mnfld_pred[valid_scaler==0]=1
            mnfld_pred = mnfld_pred.reshape(num_batch, num_point, num_init)
            mnfld_pred = torch.min(mnfld_pred, dim=-1)[0]

            smpl_verts_c = self.smpl_server.verts_c.repeat(num_batch, 1,1)
            random_idx = []
            for _ in range(num_batch):
                r = torch.randperm(num_verts).cuda()[:self.points_batch].unsqueeze(0).unsqueeze(-1)
                random_idx.append(r)
            random_idx = torch.cat(random_idx, 0).expand(-1, -1, num_dim)
            nonmnfld_pnts_c = torch.gather(smpl_verts_c, 1, random_idx)

            nonmnfld_pnts_c = self.sampler.get_points(nonmnfld_pnts_c)
            nonmnfld_pnts_c.requires_grad_()
            nonmnfld_pnts_pred_c = self.implicit_network(nonmnfld_pnts_c, cond)[..., 0:1]
            nonmnfld_grad = gradient(nonmnfld_pnts_c, nonmnfld_pnts_pred_c)

            pts_c_w, w_gt = self.sampler_bone.get_joints(self.smpl_server.joints_c.expand(1, -1, -1))
            w_pd = self.deformer.query_weights(pts_c_w, cond)

        else:

            grad_theta = None


        output = {
            'on_surface_sdf': mnfld_pred[:, :self.points_batch],
            'on_surface_gt': gt_sdf[:, :self.points_batch],
            'off_surface_sdf': mnfld_pred[:, self.points_batch:],
            'off_surface_gt': gt_sdf[:, self.points_batch:],
            'grad_theta': nonmnfld_grad,
            "w_pd": w_pd,
            "w_gt": w_gt,
        }
        return output


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

class SDF_Init(pl.LightningModule):
    def __init__(self, opt) -> None:
        super().__init__()

        # self.automatic_optimization = False
        self.model = SDF_Init_Network(opt.model)
        self.loss = ThreeDLoss(opt.model.loss)
        self.training_modules = ["model"]

        self.opt = opt

    def configure_optimizers(self):
        params = chain(*[
            getattr(self, module).parameters()
            for module in self.training_modules
        ])
        # from IPython  import embed; embed()
        # exit()
        self.optimizer = optim.Adam(params, lr=1e-3)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=[1000, 1500], gamma=0.5)
        return {"optimizer": self.optimizer, "lr_scheduler": self.scheduler}

    def training_step(self, batch, *args, **kwargs):
        inputs = batch
        model_outputs = self.model(inputs)
        loss_output = self.loss(model_outputs)
        for k, v in loss_output.items():
            if k in ["loss"]:
                self.log(k, v.item())
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)
        if self.current_epoch % 5 == 0:
            self.save_checkpoints(self.current_epoch)
        return loss_output["loss"]

    # def validation_step(self, batch, *args, **kwargs):
    #     inputs, targets = batch
    #     model_outputs = self.model(inputs)
    #     return {
    #         "rgb_values": model_outputs["rgb_values"].detach().clone(),
    #         **targets,
    #     }

    # def validation_epoch_end(self, outputs) -> None:
    #     img_size = outputs[0]["img_size"].squeeze(0)

    #     rgb_pred = torch.cat([output["rgb_values"] for output in outputs], dim=0)
    #     rgb_pred = (rgb_pred.reshape(*img_size, -1) + 1) / 2

    #     rgb_gt = torch.cat([output["rgb"] for output in outputs], dim=1).squeeze(0)
    #     rgb_gt = (rgb_gt.reshape(*img_size, -1) + 1) / 2

    #     rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
    #     rgb = (rgb * 255).astype(np.uint8)

    #     os.makedirs("rendering", exist_ok=True)
    #     cv2.imwrite(f"rendering/{self.current_epoch}.png", rgb[:, :, ::-1])

    def save_checkpoints(self, epoch):
        if not os.path.exists("checkpoints/ModelParameters"):
            os.makedirs("checkpoints/ModelParameters")
        if not os.path.exists("checkpoints/OptimizerParameters"):
            os.makedirs("checkpoints/OptimizerParameters")
        torch.save(
            {"epoch": epoch, "model_state_dict": self.model.implicit_network.state_dict(),
                'deformer_state_dict': self.model.deformer.state_dict()},
            os.path.join("checkpoints", "ModelParameters", str(epoch) + ".pth"))
        # torch.save(
        #     {"epoch": epoch, "model_state_dict": self.network.state_dict(),
        #         'deformer_state_dict': self.deformer.state_dict()},
        #     os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

        torch.save(
            {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join("checkpoints", "OptimizerParameters", str(epoch) + ".pth"))
        # torch.save(
        #     {"epoch": epoch, "optimizer_state_dict": self.optimizer.state_dict()},
        #     os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))