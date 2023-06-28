import pytorch_lightning as pl
import torch.optim as optim
from lib.model.v2a import V2A
from lib.model.body_model_params import BodyModelParams
import cv2
import torch
from lib.model.loss import Loss
import hydra
import os
import numpy as np
from lib.utils.mesh import generate_mesh
from kaolin.ops.mesh import index_vertices_by_faces
import trimesh
from lib.model.deformer import skinning
from lib.utils import idr_utils
from lib.datasets import create_dataset
from tqdm import tqdm
from lib.model.render import Renderer
from typing import Any, Callable, Dict, Generator, List, Mapping, Optional, overload, Sequence, Tuple, Union
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim.optimizer import Optimizer
from lib.model.sam_model import SAMServer
class V2AModel(pl.LightningModule):
    def __init__(self, opt, betas_path) -> None:
        super().__init__()
        self.model = V2A(opt.model, betas_path)
        self.opt = opt
        self.num_training_frames = opt.model.num_training_frames
        self.start_frame = opt.dataset.train.start_frame
        self.end_frame = opt.dataset.train.end_frame
        self.training_indices = list(range(self.start_frame, self.end_frame))
        assert len(self.training_indices) == self.num_training_frames
        self.opt_smpl = True
        self.training_modules = ["model"]
        self.num_person = opt.dataset.train.num_person
        if self.opt_smpl:
            self.body_model_list = torch.nn.ModuleList()
            for i in range(self.num_person):
                body_model_params = BodyModelParams(opt.model.num_training_frames, model_type='smpl')
                self.body_model_list.append(body_model_params)
                self.load_body_model_params(i)
                optim_params = self.body_model_list[i].param_names
                for param_name in optim_params:
                    self.body_model_list[i].set_requires_grad(param_name, requires_grad=True)
            self.training_modules += ['body_model_list']

        self.loss = Loss(opt.model.loss)
        self.sam_server = SAMServer(opt.dataset.train)
        self.using_sam = opt.dataset.train.using_SAM
        self.pose_correction_epoch = opt.model.pose_correction_epoch


        
    def load_body_model_params(self, index):
        body_model_params = {param_name: [] for param_name in self.body_model_list[index].param_names}
        data_root = os.path.join('../data', self.opt.dataset.train.data_dir)
        data_root = hydra.utils.to_absolute_path(data_root)

        body_model_params['betas'] = torch.tensor(np.load(os.path.join(data_root, 'mean_shape.npy'))[None, index], dtype=torch.float32)
        body_model_params['global_orient'] = torch.tensor(np.load(os.path.join(data_root, 'poses.npy'))[self.training_indices][:, index, :3], dtype=torch.float32)
        body_model_params['body_pose'] = torch.tensor(np.load(os.path.join(data_root, 'poses.npy'))[self.training_indices] [:, index, 3:], dtype=torch.float32)
        body_model_params['transl'] = torch.tensor(np.load(os.path.join(data_root, 'normalize_trans.npy'))[self.training_indices][:, index], dtype=torch.float32)

        for param_name in body_model_params.keys():
            self.body_model_list[index].init_parameters(param_name, body_model_params[param_name], requires_grad=False)

    def configure_optimizers(self):
        params = [{'params': self.model.parameters(), 'lr':self.opt.model.learning_rate}]
        if self.opt_smpl:
            params.append({'params': self.body_model_list.parameters(), 'lr':self.opt.model.learning_rate*0.1})
        self.optimizer = optim.Adam(params, lr=self.opt.model.learning_rate, eps=1e-8)
        # self.optimizer = optim.SGD(params, lr=self.opt.model.learning_rate*100000)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch):
        # self.get_instance_mask()
        # self.get_sam_mask()

        inputs, targets = batch

        batch_idx = inputs["idx"]
        # import ipdb;ipdb.set_trace()
        if self.using_sam:
            is_certain = inputs["is_certain"].squeeze()
            # if self.current_epoch < 500 and batch_idx > 85 and batch_idx < 125: # this is for warmwelcome
            if self.current_epoch < self.pose_correction_epoch and not is_certain:
            # if self.current_epoch < 500 and batch_idx > 38 and batch_idx < 106: # this is for piggyback
                for param in self.model.foreground_implicit_network_list.parameters():
                    param.requires_grad = False
                for param in self.model.foreground_rendering_network_list.parameters():
                    param.requires_grad = False
                for param in self.model.bg_implicit_network.parameters():
                    param.requires_grad = False
                for param in self.model.bg_rendering_network.parameters():
                    param.requires_grad = False
            else:
                for param in self.model.foreground_implicit_network_list.parameters():
                    param.requires_grad = True
                for param in self.model.foreground_rendering_network_list.parameters():
                    param.requires_grad = True
                for param in self.model.bg_implicit_network.parameters():
                    param.requires_grad = True
                for param in self.model.bg_rendering_network.parameters():
                    param.requires_grad = True
        
        device = inputs["smpl_params"].device

        if self.opt_smpl:
            # body_model_params = self.body_model_params(batch_idx)
            body_params_list = [self.body_model_list[i](batch_idx) for i in range(self.num_person)]
            inputs['smpl_trans'] = torch.stack([body_model_params['transl'] for body_model_params in body_params_list], dim=1)
            inputs['smpl_shape'] = torch.stack([body_model_params['betas'] for body_model_params in body_params_list], dim=1)
            global_orient = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list], dim=1)
            body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list], dim=1)
            inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)

            if batch_idx == 0:
                last_idx = batch_idx
            else:
                last_idx = batch_idx - 1
            body_params_list_last = [self.body_model_list[i](last_idx) for i in range(self.num_person)]
            global_orient_last = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list_last],
                                        dim=1)
            body_pose_last = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list_last], dim=1)
            inputs['smpl_pose_last'] = torch.cat((global_orient_last, body_pose_last), dim=2)
            # inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
            # inputs['smpl_shape'] = body_model_params['betas']
            # inputs['smpl_trans'] = body_model_params['transl']
        else:
            # inputs['smpl_pose'] = inputs["smpl_params"][:, 4:76]
            # inputs['smpl_shape'] = inputs["smpl_params"][:, 76:]
            # inputs['smpl_trans'] = inputs["smpl_params"][:, 1:4]
            inputs['smpl_pose'] = inputs["smpl_params"][..., 4:76]
            inputs['smpl_shape'] = inputs["smpl_params"][..., 76:]
            inputs['smpl_trans'] = inputs["smpl_params"][..., 1:4]

        inputs['current_epoch'] = self.current_epoch
        # import pdb;pdb.set_trace()
        model_outputs = self.model(inputs)

        loss_output = self.loss(model_outputs, targets)
        for k, v in loss_output.items():
            if k in ["loss"]:
                self.log(k, v.item(), prog_bar=True, on_step=True)
            else:
                self.log(k, v.item(), prog_bar=True, on_step=True)
        return loss_output["loss"]

    # def backward(
    #     self, loss, optimizer, optimizer_idx, *args, **kwargs
    # ) -> None:
    #     """Called to perform backward on the loss returned in :meth:`training_step`. Override this hook with your
    #     own implementation if you need to.
    #
    #     Args:
    #         loss: The loss tensor returned by :meth:`training_step`. If gradient accumulation is used, the loss here
    #             holds the normalized value (scaled by 1 / accumulation steps).
    #         optimizer: Current optimizer being used. ``None`` if using manual optimization.
    #         optimizer_idx: Index of the current optimizer being used. ``None`` if using manual optimization.
    #
    #     Example::
    #
    #         def backward(self, loss, optimizer, optimizer_idx):
    #             loss.backward()
    #     """
    #     if self._fabric:
    #         self._fabric.backward(loss, *args, **kwargs)
    #     else:
    #         loss.backward(*args, **kwargs)
    #     import ipdb; ipdb.set_trace()

    # def optimizer_step(
    #     self,
    #     epoch: int,
    #     batch_idx: int,
    #     optimizer: Union[Optimizer, LightningOptimizer],
    #     optimizer_idx: int = 0,
    #     optimizer_closure: Optional[Callable[[], Any]] = None,
    #     on_tpu: bool = False,
    #     using_lbfgs: bool = False,
    # ) -> None:
    #     r"""
    #     Override this method to adjust the default way the :class:`~pytorch_lightning.trainer.trainer.Trainer` calls
    #     each optimizer.
    #
    #     By default, Lightning calls ``step()`` and ``zero_grad()`` as shown in the example once per optimizer.
    #     This method (and ``zero_grad()``) won't be called during the accumulation phase when
    #     ``Trainer(accumulate_grad_batches != 1)``. Overriding this hook has no benefit with manual optimization.
    #
    #     Args:
    #         epoch: Current epoch
    #         batch_idx: Index of current batch
    #         optimizer: A PyTorch optimizer
    #         optimizer_idx: If you used multiple optimizers, this indexes into that list.
    #         optimizer_closure: The optimizer closure. This closure must be executed as it includes the
    #             calls to ``training_step()``, ``optimizer.zero_grad()``, and ``backward()``.
    #         on_tpu: ``True`` if TPU backward is required
    #         using_lbfgs: True if the matching optimizer is :class:`torch.optim.LBFGS`
    #
    #     Examples::
    #
    #         # DEFAULT
    #         def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                            optimizer_closure, on_tpu, using_lbfgs):
    #             optimizer.step(closure=optimizer_closure)
    #
    #         # Alternating schedule for optimizer steps (i.e.: GANs)
    #         def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
    #                            optimizer_closure, on_tpu, using_lbfgs):
    #             # update generator opt every step
    #             if optimizer_idx == 0:
    #                 optimizer.step(closure=optimizer_closure)
    #
    #             # update discriminator opt every 2 steps
    #             if optimizer_idx == 1:
    #                 if (batch_idx + 1) % 2 == 0 :
    #                     optimizer.step(closure=optimizer_closure)
    #                 else:
    #                     # call the closure by itself to run `training_step` + `backward` without an optimizer step
    #                     optimizer_closure()
    #
    #             # ...
    #             # add as many optimizers as you want
    #
    #     Here's another example showing how to use this for more advanced things such as
    #     learning rate warm-up:
    #
    #     .. code-block:: python
    #
    #         # learning rate warm-up
    #         def optimizer_step(
    #             self,
    #             epoch,
    #             batch_idx,
    #             optimizer,
    #             optimizer_idx,
    #             optimizer_closure,
    #             on_tpu,
    #             using_lbfgs,
    #         ):
    #             # update params
    #             optimizer.step(closure=optimizer_closure)
    #
    #             # manually warm up lr without a scheduler
    #             if self.trainer.global_step < 500:
    #                 lr_scale = min(1.0, float(self.trainer.global_step + 1) / 500.0)
    #                 for pg in optimizer.param_groups:
    #                     pg["lr"] = lr_scale * self.learning_rate
    #
    #     """
    #     import ipdb; ipdb.set_trace()
    #     optimizer.step(closure=optimizer_closure)

    def training_epoch_end(self, outputs) -> None:
        # Canonical mesh update every 20 epochs
        if self.current_epoch != 0 and self.current_epoch % 20 == 0:
            for person_id, smpl_server in enumerate(self.model.smpl_server_list):
                # cond = {'smpl': torch.zeros(1, 69).float().cuda()}
                cond_pose = torch.zeros(1, 69).float().cuda()
                if self.model.use_person_encoder:
                    # import pdb;pdb.set_trace()
                    person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(cond_pose.device)
                    person_encoding = self.model.person_latent_encoder(person_id_tensor)
                    person_encoding = person_encoding.repeat(cond_pose.shape[0], 1)
                    cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                    cond = {'smpl_id': cond_pose_id}
                else:
                    cond = {'smpl': cond_pose}
                mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_id), smpl_server.verts_c[0], point_batch=10000, res_up=2)
                self.model.mesh_v_cano_list[person_id] = torch.tensor(mesh_canonical.vertices[None], device = self.model.mesh_v_cano_list[person_id].device).float()
                self.model.mesh_f_cano_list[person_id] = torch.tensor(mesh_canonical.faces.astype(np.int64), device=self.model.mesh_v_cano_list[person_id].device)
                self.model.mesh_face_vertices_list[person_id] = index_vertices_by_faces(self.model.mesh_v_cano_list[person_id], self.model.mesh_f_cano_list[person_id])
        if self.current_epoch % 50 == 0:
            self.get_instance_mask()
            self.get_sam_mask()
        return super().training_epoch_end(outputs)

    def get_sam_mask(self):
        print("start get refined SAM mask")
        self.sam_server.get_sam_mask(self.current_epoch)
    def get_instance_mask(self):
        print("start get SMPL instance mask")
        self.model.eval()
        os.makedirs(f"stage_mask/{self.current_epoch:05d}/all", exist_ok=True)
        os.makedirs(f"stage_rendering/{self.current_epoch:05d}/all", exist_ok=True)
        os.makedirs(f"stage_fg_rendering/{self.current_epoch:05d}/all", exist_ok=True)
        os.makedirs(f"stage_normal/{self.current_epoch:05d}/all", exist_ok=True)
        # os.makedirs(f"stage_mesh/{self.current_epoch:05d}/all", exist_ok=True)
        testset = create_dataset(self.opt.dataset.test)
        keypoint_list = [[] for _ in range(len(self.model.smpl_server_list))]
        all_person_smpl_mask_list =[]
        all_instance_mask_depth_list = []
        for batch_ndx, batch in enumerate(tqdm(testset)):
            # generate instance mask for all test images
            inputs, targets, pixel_per_batch, total_pixels, idx = batch
            inputs = {key: value.cuda() for key, value in inputs.items()}
            # targets = {key: value.cuda() for key, value in targets.items()}
            num_splits = (total_pixels + pixel_per_batch -
                          1) // pixel_per_batch
            results = []
            for i in range(num_splits):
                indices = list(range(i * pixel_per_batch,
                                     min((i + 1) * pixel_per_batch, total_pixels)))
                batch_inputs = {"uv": inputs["uv"][:, indices],
                                "P": inputs["P"],
                                "C": inputs["C"],
                                "intrinsics": inputs['intrinsics'],
                                "pose": inputs['pose'],
                                "smpl_params": inputs["smpl_params"],
                                "smpl_pose": inputs["smpl_params"][:, :, 4:76],
                                "smpl_shape": inputs["smpl_params"][:, :, 76:],
                                "smpl_trans": inputs["smpl_params"][:, :, 1:4],
                                "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

                if self.opt_smpl:
                    body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
                    batch_inputs['smpl_trans'] = torch.stack(
                        [body_model_params['transl'] for body_model_params in body_params_list],
                        dim=1)
                    batch_inputs['smpl_shape'] = torch.stack(
                        [body_model_params['betas'] for body_model_params in body_params_list],
                        dim=1)
                    global_orient = torch.stack(
                        [body_model_params['global_orient'] for body_model_params in body_params_list],
                        dim=1)
                    body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list],
                                            dim=1)
                    batch_inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)

                batch_targets = {
                    "rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                    "img_size": targets["img_size"]}
                # here -1 means enable all human VolSDF

                # model_outputs = self.model(batch_inputs, -1)
                # results.append({"rgb_values": model_outputs["rgb_values"].detach().clone(),
                #                 "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                #                 "normal_values": model_outputs["normal_values"].detach().clone(),
                #                 "acc_map": model_outputs["acc_map"].detach().clone(),
                #                 "acc_person_list": model_outputs["acc_person_list"].detach().clone(),
                #                 **batch_targets})
                results.append({**batch_targets})

            img_size = targets["img_size"]
            P = batch_inputs["P"][0].cpu().numpy()
            P_norm = np.eye(4)
            P_norm[:, :] = P[:, :]
            assert batch_inputs["smpl_params"][:, 0, 0] == batch_inputs["smpl_params"][:, 1, 0]
            scale = batch_inputs["smpl_params"][:, 0, 0]
            scale_eye = np.eye(4)
            scale_eye[0, 0] = scale
            scale_eye[1, 1] = scale
            scale_eye[2, 2] = scale
            P_norm = P_norm @ scale_eye
            # 其实蛮奇怪的，最后一维的偏移没有乘以scale，也就是说（P_norm，1/scale * vert）和 （P，verts）不完全等效？
            # TODO camera pose P is not correct  after scale, may influence shooting the ray
            out = cv2.decomposeProjectionMatrix(P_norm[:3, :])
            cam_intrinsics = out[0]
            render_R = out[1]
            cam_center = out[2]
            cam_center = (cam_center[:3] / cam_center[3])[:, 0]
            render_T = -render_R @ cam_center
            render_R = torch.tensor(render_R)[None].float()
            render_T = torch.tensor(render_T)[None].float()
            renderer = Renderer(img_size=[img_size[0], img_size[1]],
                                cam_intrinsic=cam_intrinsics)

            img_size = targets["img_size"]
            # renderer = Renderer(img_size=[img_size[0], img_size[1]], cam_intrinsic=batch_inputs["intrinsics"][0].cpu().numpy())
            # pose = batch_inputs["pose"][0].cpu().numpy()
            # render_R = torch.tensor(pose[:3, :3])[None].float()
            # render_T = torch.tensor(pose[:3, 3])[None].float()
            renderer.set_camera(render_R, render_T)
            verts_list = []
            faces_list = []
            colors_list = []
            color_dict = [[255, 0.0, 0.0], [0.0, 255, 0.0]]
            for person_idx, smpl_server in enumerate(self.model.smpl_server_list):
                smpl_outputs = smpl_server(batch_inputs["smpl_params"][:, person_idx, 0], batch_inputs['smpl_trans'][:, person_idx], batch_inputs['smpl_pose'][:, person_idx], batch_inputs['smpl_shape'][:, person_idx])
                # smpl_outputs["smpl_verts"] shape (1, 6890, 3)
                verts_color = torch.tensor(color_dict[person_idx]).repeat(smpl_outputs["smpl_verts"].shape[1], 1)
                # import pdb;pdb.set_trace()
                # print(verts_color.shape)

                # here we invert the scale back!!!!!
                smpl_mesh = trimesh.Trimesh((1/scale.squeeze().detach().cpu()) * smpl_outputs["smpl_verts"].squeeze().detach().cpu(), smpl_server.smpl.faces, process=False, vertex_colors=verts_color.cpu())
                verts = torch.tensor(smpl_mesh.vertices).cuda().float()[None]
                faces = torch.tensor(smpl_mesh.faces).cuda()[None]
                colors = torch.tensor(smpl_mesh.visual.vertex_colors).float().cuda()[None,..., :3] / 255

                verts_list.append(verts)
                faces_list.append(faces)
                colors_list.append(colors)
                P = batch_inputs["P"][0].cpu().numpy()
                smpl_joints = smpl_outputs["smpl_all_jnts"].detach().cpu().numpy().squeeze()
                # print(smpl_joints.shape)
                # exit()
                smpl_joints = smpl_joints[:27]  # original smpl point + nose + eyes
                # smpl_joints = smpl_outputs["smpl_verts"].squeeze().detach().cpu().numpy()
                pix_list = []
                # get the ground truth image
                img_size = results[0]["img_size"]
                rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
                input_img = rgb_gt.reshape(*img_size, -1).detach().cpu().numpy()
                input_img = (input_img * 255).astype(np.uint8)

                for j in range(0, smpl_joints.shape[0]):
                    padded_v = np.pad(smpl_joints[j], (0, 1), 'constant', constant_values=(0, 1))
                    temp = P @ padded_v.T  # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
                    pix = (temp / temp[2])[:2]
                    output_img = cv2.circle(input_img, tuple(pix.astype(np.int32)), 3, (0, 255, 255), -1)
                    pix_list.append(pix.astype(np.int32))
                pix_tensor = np.stack(pix_list, axis=0)
                keypoint_list[person_idx].append(pix_tensor)
                # if not os.path.exists(f'stage_joint_opt_smpl_joint/{person_idx}'):
                #     os.makedirs(f'stage_joint_opt_smpl_joint/{person_idx}')
                # cv2.imwrite(os.path.join(f'stage_joint_opt_smpl_joint/{person_idx}', '%04d.png' % idx), output_img[:, :, ::-1])
            renderer_depth_map = renderer.render_multiple_depth_map(verts_list, faces_list, colors_list)
            renderer_smpl_img = renderer.render_multiple_meshes(verts_list, faces_list, colors_list)
            renderer_smpl_img = (255 * renderer_smpl_img).data.cpu().numpy().astype(np.uint8)
            renderer_smpl_img = renderer_smpl_img[0]
            img_size = results[0]["img_size"]
            rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            input_img = rgb_gt.reshape(*img_size, -1).detach().cpu().numpy()
            input_img = (input_img * 255).astype(np.uint8)
            if input_img.shape[0] < input_img.shape[1]:
                renderer_smpl_img = renderer_smpl_img[abs(input_img.shape[0] - input_img.shape[1]) // 2:(input_img.shape[0] + input_img.shape[1]) // 2, ...]
            else:
                renderer_smpl_img = renderer_smpl_img[:, abs(input_img.shape[0] - input_img.shape[1]) // 2:(input_img.shape[0] + input_img.shape[1]) // 2]
            reshape_depth_map_list = []
            for map_id, depth_map_i in enumerate(renderer_depth_map):
                depth_map_i = depth_map_i[0,:,:,0].data.cpu().numpy()
                if input_img.shape[0] < input_img.shape[1]:
                    depth_map_i = depth_map_i[abs(input_img.shape[0] - input_img.shape[1]) // 2:(input_img.shape[0] + input_img.shape[1]) // 2, ...]
                else:
                    depth_map_i = depth_map_i[:, abs(input_img.shape[0] - input_img.shape[1]) // 2:(input_img.shape[0] + input_img.shape[1]) // 2]
                reshape_depth_map_list.append(depth_map_i)
            # get front depth map
            max_depth_map_list = []
            for map_id, depth_map_i in enumerate(reshape_depth_map_list):
                depth_map_processed = np.copy(depth_map_i)
                no_interaction = depth_map_processed < 0
                max_depth = 999
                depth_map_processed[no_interaction] = max_depth
                max_depth_map_list.append(depth_map_processed)
            max_depth_map = np.stack(max_depth_map_list, axis=0)
            front_depth_map = np.min(max_depth_map, axis=0)
            # get instance mask from depth map
            instance_mask_list = []
            for map_id, depth_map_i in enumerate(reshape_depth_map_list):
                instance_mask = (depth_map_i == front_depth_map)
                instance_mask_list.append(instance_mask)
                all_red_image = np.ones((instance_mask.shape[0], instance_mask.shape[1], 3)) * np.array([255, 0, 0]).reshape(1, 1, 3)
                instance_mask = instance_mask[:, :, np.newaxis]
                output_img_person_1 = (all_red_image * instance_mask + input_img * (1 - instance_mask)).astype(np.uint8)
                os.makedirs(f"stage_depth_instance_mask/{self.current_epoch:05d}", exist_ok=True)
                cv2.imwrite(os.path.join(f"stage_depth_instance_mask/{self.current_epoch:05d}",
                                         f'{map_id}_smpl_render_%04d.png' % idx), output_img_person_1[:, :, ::-1])
            all_instance_mask_depth = np.stack(instance_mask_list, axis=0)
            all_instance_mask_depth_list.append(all_instance_mask_depth)


            for map_id, depth_map_i in enumerate(reshape_depth_map_list):
                # Processing depth map for better visualization
                depth_map_processed = np.copy(depth_map_i)

                # Assigning no interaction areas (-1s) to max value for visualization
                no_interaction = depth_map_processed < 0
                # depth_map_processed[no_interaction] = np.max(depth_map_processed[~no_interaction])
                min_depth = 2.5
                max_depth = 5
                depth_map_processed[no_interaction] = max_depth
                depth_map_processed = np.clip(depth_map_processed, min_depth, max_depth)
                # do the min max normalization manually
                depth_map_processed = (depth_map_processed - min_depth) / (max_depth - min_depth)
                depth_map_processed = (depth_map_processed * 255).astype(np.uint8)

                # Normalize the depth map to 0-255 for visual effect as image (assuming 8 bit depth)
                # depth_map_processed = cv2.normalize(depth_map_processed, None, 255, 0, norm_type=cv2.NORM_MINMAX,
                #                                     dtype=cv2.CV_8U)

                # Apply the reversed 'JET' colormap
                depth_map_processed = cv2.applyColorMap(255 - depth_map_processed, cv2.COLORMAP_JET)
                os.makedirs(f"stage_depth_map/{self.current_epoch:05d}", exist_ok=True)
                cv2.imwrite(os.path.join(f"stage_depth_map/{self.current_epoch:05d}",
                                         f'{map_id}_smpl_render_%04d.png' % idx), depth_map_processed)

            valid_mask = (renderer_smpl_img[:, :, -1] > 0)[:, :, np.newaxis]

            valid_render_image = renderer_smpl_img[:, :, :-1] * valid_mask
            person_1_mask = (valid_render_image[:, :, 0] >= 250) & (valid_render_image[:, :, 0] >= valid_render_image[:, :, 1])
            person_2_mask = (valid_render_image[:, :, 1] >= 250) & (valid_render_image[:, :, 1] >= valid_render_image[:, :, 0])
            all_person_mask = np.stack([person_1_mask, person_2_mask], axis=0)
            all_person_smpl_mask_list.append(all_person_mask)
            person_1_mask = person_1_mask[:, :, np.newaxis]
            person_2_mask = person_2_mask[:, :, np.newaxis]
            all_red_image = np.ones_like(valid_render_image) * np.array([255, 0, 0]).reshape(1, 1, 3)
            output_img_person_1 = (all_red_image * person_1_mask + input_img * (1 - person_1_mask)).astype(np.uint8)
            output_img_person_2 = (all_red_image * person_2_mask + input_img * (1 - person_2_mask)).astype(np.uint8)
            # output_img = (renderer_smpl_img[:, :, :-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
            # cv2.imwrite(os.path.join(f'stage_joint_opt_smpl_joint', 'smpl_render_%04d.png' % idx), output_img[:,:,::-1])
            os.makedirs(f"stage_joint_opt_smpl_joint/{self.current_epoch:05d}", exist_ok=True)
            cv2.imwrite(os.path.join(f"stage_joint_opt_smpl_joint/{self.current_epoch:05d}", '1_smpl_render_%04d.png' % idx), output_img_person_1[:,:,::-1])
            cv2.imwrite(os.path.join(f"stage_joint_opt_smpl_joint/{self.current_epoch:05d}", '2_smpl_render_%04d.png' % idx), output_img_person_2[:,:,::-1])
            # import pdb; pdb.set_trace()
            # img_size = results[0]["img_size"]
            # rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
            # rgb_pred = rgb_pred.reshape(*img_size, -1)
            #
            # fg_rgb_pred = torch.cat([result["fg_rgb_values"] for result in results], dim=0)
            # fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)
            #
            # normal_pred = torch.cat([result["normal_values"] for result in results], dim=0)
            # normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2
            #
            # pred_mask = torch.cat([result["acc_map"] for result in results], dim=0)
            # pred_mask = pred_mask.reshape(*img_size, -1)
            #
            # instance_mask = torch.cat([result["acc_person_list"] for result in results], dim=0)
            # instance_mask = instance_mask.reshape(*img_size, -1)
            # for i in range(instance_mask.shape[2]):
            #     instance_mask_i = instance_mask[:, :, i]
            #     os.makedirs(f"stage_instance_mask/{self.current_epoch:05d}/{i}", exist_ok=True)
            #     cv2.imwrite(f"stage_instance_mask/{self.current_epoch:05d}/{i}/{int(idx.cpu().numpy()):04d}.png",
            #                 instance_mask_i.cpu().numpy() * 255)
            #
            # if results[0]['rgb'] is not None:
            #     rgb_pred = rgb_pred.cpu()
            #     rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0)
            #     rgb_gt = rgb_gt.reshape(*img_size, -1)
            #     rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
            # else:
            #     rgb = torch.cat([rgb_pred], dim=0).cpu().numpy()
            # if 'normal' in results[0].keys():
            #     normal_pred = normal_pred.cpu()
            #     normal_gt = torch.cat([result["normal"] for result in results], dim=1).squeeze(0)
            #     normal_gt = (normal_gt.reshape(*img_size, -1) + 1) / 2
            #     normal = torch.cat([normal_gt, normal_pred], dim=0).cpu().numpy()
            # else:
            #     normal = torch.cat([normal_pred], dim=0).cpu().numpy()
            #
            # rgb = (rgb * 255).astype(np.uint8)
            #
            # fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
            # fg_rgb = (fg_rgb * 255).astype(np.uint8)
            #
            # normal = (normal * 255).astype(np.uint8)
            #
            # cv2.imwrite(f"stage_mask/{self.current_epoch:05d}/all/{int(idx.cpu().numpy()):04d}.png", pred_mask.cpu().numpy() * 255)
            # cv2.imwrite(f"stage_rendering/{self.current_epoch:05d}/all/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
            # cv2.imwrite(f"stage_normal/{self.current_epoch:05d}/all/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])
            # cv2.imwrite(f"stage_fg_rendering/{self.current_epoch:05d}/all/{int(idx.cpu().numpy()):04d}.png", fg_rgb[:, :, ::-1])
        all_instance_mask_depth_list = np.array(all_instance_mask_depth_list)
        all_person_smpl_mask_list = np.array(all_person_smpl_mask_list)
        print("all_person_smpl_mask_list.shape ", all_person_smpl_mask_list.shape)
        print("all_instance_mask_depth_list.shape ", all_instance_mask_depth_list.shape)
        keypoint_list = np.array(keypoint_list)
        keypoint_list = keypoint_list.transpose(1, 0, 2, 3)
        os.makedirs(f"stage_instance_mask/{self.current_epoch:05d}", exist_ok=True)
        # np.save(f'stage_instance_mask/{self.current_epoch:05d}/all_person_smpl_mask.npy', all_person_smpl_mask_list)
        np.save(f'stage_instance_mask/{self.current_epoch:05d}/all_person_smpl_mask.npy', all_instance_mask_depth_list)
        np.save(f'stage_instance_mask/{self.current_epoch:05d}/2d_keypoint.npy', keypoint_list)
        # shape (160, 2, 960, 540)
        print("all_person_smpl_mask_list.shape ", all_person_smpl_mask_list.shape)
        # shape (160, 2, 27, 2)
        print("keypoint_list.shape ", keypoint_list.shape)
        self.model.train()

    def query_oc(self, x, cond, person_id):
        
        x = x.reshape(-1, 3)
        mnfld_pred = self.model.foreground_implicit_network_list[person_id](x, cond)[:,:,0].reshape(-1,1)
        return {'occ':mnfld_pred}

    def query_wc(self, x):
        
        x = x.reshape(-1, 3)
        w = self.model.deformer.query_weights(x)
    
        return w

    def query_od(self, x, cond, smpl_tfs, smpl_verts):
        
        x = x.reshape(-1, 3)
        x_c, _ = self.model.deformer.forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
        output = self.model.implicit_network(x_c, cond)[0]
        sdf = output[:, 0:1]
        
        return {'occ': sdf}

    def get_deformed_mesh_fast_mode(self, verts, smpl_tfs):
        verts = torch.tensor(verts).cuda().float()
        weights = self.model.deformer.query_weights(verts)
        verts_deformed = skinning(verts.unsqueeze(0),  weights, smpl_tfs).data.cpu().numpy()[0]
        return verts_deformed

    def validation_step(self, batch, *args, **kwargs):
        outputs = []
        outputs.append(self.validation_step_single_person(batch, id=-1))
        if self.num_person > 1:
            for i in range(self.num_person):
                outputs.append(self.validation_step_single_person(batch, id=i))

        return outputs


    def validation_step_single_person(self, batch, id):

        output = {}
        inputs, targets = batch
        inputs['current_epoch'] = self.current_epoch
        self.model.eval()

        device = inputs["smpl_params"].device
        # if self.opt_smpl:
        #     body_model_params = self.body_model_params(inputs['image_id'])
        #     inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
        #     inputs['smpl_shape'] = body_model_params['betas']
        #     inputs['smpl_trans'] = body_model_params['transl']
        #
        # else:
        #     inputs['smpl_pose'] = inputs["smpl_params"][:, 4:76]
        #     inputs['smpl_shape'] = inputs["smpl_params"][:, 76:]
        #     inputs['smpl_trans'] = inputs["smpl_params"][:, 1:4]
        if self.opt_smpl:
            # body_model_params = self.body_model_params(batch_idx)
            body_params_list = [self.body_model_list[i](inputs['image_id']) for i in range(self.num_person)]
            inputs['smpl_trans'] = torch.stack([body_model_params['transl'] for body_model_params in body_params_list],
                                               dim=1)
            inputs['smpl_shape'] = torch.stack([body_model_params['betas'] for body_model_params in body_params_list],
                                               dim=1)
            global_orient = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list],
                                        dim=1)
            body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list], dim=1)
            inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)
            # inputs['smpl_pose'] = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
            # inputs['smpl_shape'] = body_model_params['betas']
            # inputs['smpl_trans'] = body_model_params['transl']
        else:
            # inputs['smpl_pose'] = inputs["smpl_params"][:, 4:76]
            # inputs['smpl_shape'] = inputs["smpl_params"][:, 76:]
            # inputs['smpl_trans'] = inputs["smpl_params"][:, 1:4]
            inputs['smpl_pose'] = inputs["smpl_params"][..., 4:76]
            inputs['smpl_shape'] = inputs["smpl_params"][..., 76:]
            inputs['smpl_trans'] = inputs["smpl_params"][..., 1:4]

        mesh_canonical_list = []
        if id == -1:
            for person_id, smpl_server in enumerate(self.model.smpl_server_list):
                # cond = {'smpl': inputs["smpl_pose"][:, person_id, 3:] / np.pi}

                cond_pose = inputs["smpl_pose"][:, person_id, 3:] / np.pi
                if self.model.use_person_encoder:
                    # import pdb;pdb.set_trace()
                    person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(inputs["smpl_pose"].device)
                    person_encoding = self.model.person_latent_encoder(person_id_tensor)
                    person_encoding = person_encoding.repeat(inputs["smpl_pose"].shape[0], 1)
                    cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                    cond = {'smpl_id': cond_pose_id}
                else:
                    cond = {'smpl': cond_pose}

                # mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0], point_batch=10000, res_up=3)
                mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_id),
                                               smpl_server.verts_c[0], point_batch=10000, res_up=3)
                mesh_canonical = trimesh.Trimesh(mesh_canonical.vertices, mesh_canonical.faces)
                mesh_canonical_list.append(mesh_canonical)

        output.update({
            'canonical_weighted': mesh_canonical_list
        })

        split = idr_utils.split_input(inputs, targets["total_pixels"][0], n_pixels=min(targets['pixel_per_batch'],
                                                                                       targets["img_size"][0] *
                                                                                       targets["img_size"][1]))

        res = []
        for s in split:
            if id==-1:
                out = self.model(s)
            else:
                out = self.model(s, id)

            for k, v in out.items():
                try:
                    out[k] = v.detach()
                except:
                    out[k] = v

            res.append({
                'rgb_values': out['rgb_values'].detach(),
                'normal_values': out['normal_values'].detach(),
                'fg_rgb_values': out['fg_rgb_values'].detach(),
            })
        batch_size = targets['rgb'].shape[0]

        model_outputs = idr_utils.merge_output(res, targets["total_pixels"][0], batch_size)

        output.update({
            "rgb_values": model_outputs["rgb_values"].detach().clone(),
            "normal_values": model_outputs["normal_values"].detach().clone(),
            "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
            **targets,
        })
        return output

    def validation_step_end(self, batch_parts):
        return batch_parts

    def validation_epoch_end_person(self, outputs, person_id):
        img_size = outputs[0]["img_size"]

        rgb_pred = torch.cat([output["rgb_values"] for output in outputs], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([output["fg_rgb_values"] for output in outputs], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

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

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        os.makedirs("rendering", exist_ok=True)
        os.makedirs("normal", exist_ok=True)
        os.makedirs('fg_rendering', exist_ok=True)

        canonical_mesh_list = outputs[0]['canonical_weighted']
        for i, canonical_mesh in enumerate(canonical_mesh_list):
            canonical_mesh.export(f"rendering/{self.current_epoch}_{i}.ply")

        cv2.imwrite(f"rendering/{self.current_epoch}_person{person_id}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"normal/{self.current_epoch}_person{person_id}.png", normal[:, :, ::-1])
        cv2.imwrite(f"fg_rendering/{self.current_epoch}_person{person_id}.png", fg_rgb[:, :, ::-1])

    def validation_epoch_end(self, outputs) -> None:
        # import pdb; pdb.set_trace()
        if self.num_person < 2:
            self.validation_epoch_end_person([outputs[0][0]], person_id=-1)
        else:
            self.validation_epoch_end_person([outputs[0][0]], person_id=-1)
            for i in range(self.num_person):
                self.validation_epoch_end_person([outputs[0][i+1]], person_id=i)
    
    def test_step_each_person(self, batch, id):
        os.makedirs(f"test_mask/{id}", exist_ok=True)
        os.makedirs(f"test_rendering/{id}", exist_ok=True)
        os.makedirs(f"test_fg_rendering/{id}", exist_ok=True)
        os.makedirs(f"test_normal/{id}", exist_ok=True)
        os.makedirs(f"test_mesh/{id}", exist_ok=True)

        inputs, targets, pixel_per_batch, total_pixels, idx = batch
        num_splits = (total_pixels + pixel_per_batch -
                      1) // pixel_per_batch
        results = []

        scale, smpl_trans, smpl_pose, smpl_shape = torch.split(inputs["smpl_params"], [1, 3, 72, 10], dim=2)

        if self.opt_smpl:
            # body_model_params = self.body_model_params(inputs['idx'])
            # smpl_shape = body_model_params['betas'] if body_model_params['betas'].dim() == 2 else body_model_params[
            #     'betas'].unsqueeze(0)
            # smpl_trans = body_model_params['transl']
            # smpl_pose = torch.cat((body_model_params['global_orient'], body_model_params['body_pose']), dim=1)
            body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
            smpl_trans = torch.stack([body_model_params['transl'] for body_model_params in body_params_list],
                                               dim=1)
            smpl_shape = torch.stack([body_model_params['betas'] for body_model_params in body_params_list],
                                               dim=1)
            global_orient = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list],
                                        dim=1)
            body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list], dim=1)
            smpl_pose = torch.cat((global_orient, body_pose), dim=2)

        # smpl_outputs = self.model.smpl_server(scale, smpl_trans, smpl_pose, smpl_shape)
        # smpl_tfs = smpl_outputs['smpl_tfs']
        # smpl_verts = smpl_outputs['smpl_verts']
        # cond = {'smpl': smpl_pose[:, 3:] / np.pi}
        #
        # mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond), self.model.smpl_server.verts_c[0],
        #                                point_batch=10000, res_up=4)
        # verts_deformed = self.get_deformed_mesh_fast_mode(mesh_canonical.vertices, smpl_tfs)
        # mesh_deformed = trimesh.Trimesh(vertices=verts_deformed, faces=mesh_canonical.faces, process=False)
        #
        #
        # mesh_canonical.export(f"test_mesh/{int(idx.cpu().numpy()):04d}_canonical.ply")
        # mesh_deformed.export(f"test_mesh/{int(idx.cpu().numpy()):04d}_deformed.ply")

        for i in range(num_splits):
            indices = list(range(i * pixel_per_batch,
                                 min((i + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"uv": inputs["uv"][:, indices],
                            "P": inputs["P"],
                            "C": inputs["C"],
                            "intrinsics": inputs['intrinsics'],
                            "pose": inputs['pose'],
                            "smpl_params": inputs["smpl_params"],
                            "smpl_pose": inputs["smpl_params"][:,: , 4:76],
                            "smpl_shape": inputs["smpl_params"][:,: , 76:],
                            "smpl_trans": inputs["smpl_params"][:,:, 1:4],
                            "idx": inputs["idx"] if 'idx' in inputs.keys() else None}

            if self.opt_smpl:
                # body_model_params = self.body_model_params(inputs['idx'])
                #
                # batch_inputs.update({'smpl_pose': torch.cat(
                #     (body_model_params['global_orient'], body_model_params['body_pose']), dim=1)})
                # batch_inputs.update({'smpl_shape': body_model_params['betas']})
                # batch_inputs.update({'smpl_trans': body_model_params['transl']})
                body_params_list = [self.body_model_list[i](inputs['idx']) for i in range(self.num_person)]
                batch_inputs['smpl_trans'] = torch.stack(
                    [body_model_params['transl'] for body_model_params in body_params_list],
                    dim=1)
                batch_inputs['smpl_shape'] = torch.stack(
                    [body_model_params['betas'] for body_model_params in body_params_list],
                    dim=1)
                global_orient = torch.stack(
                    [body_model_params['global_orient'] for body_model_params in body_params_list],
                    dim=1)
                body_pose = torch.stack([body_model_params['body_pose'] for body_model_params in body_params_list],
                                        dim=1)
                batch_inputs['smpl_pose'] = torch.cat((global_orient, body_pose), dim=2)

            batch_targets = {"rgb": targets["rgb"][:, indices].detach().clone() if 'rgb' in targets.keys() else None,
                             "img_size": targets["img_size"]}

            # with torch.no_grad():
            with torch.inference_mode(mode=False):
                batch_clone = {key: value.clone() for key, value in batch_inputs.items()}
                model_outputs = self.model(batch_clone, id)
            results.append({"rgb_values": model_outputs["rgb_values"].detach().clone(),
                            "fg_rgb_values": model_outputs["fg_rgb_values"].detach().clone(),
                            "normal_values": model_outputs["normal_values"].detach().clone(),
                            "acc_map": model_outputs["acc_map"].detach().clone(),
                            "acc_person_list": model_outputs["acc_person_list"].detach().clone(),
                            **batch_targets})

        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = rgb_pred.reshape(*img_size, -1)

        fg_rgb_pred = torch.cat([result["fg_rgb_values"] for result in results], dim=0)
        fg_rgb_pred = fg_rgb_pred.reshape(*img_size, -1)

        normal_pred = torch.cat([result["normal_values"] for result in results], dim=0)
        normal_pred = (normal_pred.reshape(*img_size, -1) + 1) / 2

        pred_mask = torch.cat([result["acc_map"] for result in results], dim=0)
        pred_mask = pred_mask.reshape(*img_size, -1)

        if id==-1:
            instance_mask = torch.cat([result["acc_person_list"] for result in results], dim=0)
            instance_mask = instance_mask.reshape(*img_size, -1)
            for i in range(instance_mask.shape[2]):
                instance_mask_i = instance_mask[:, :, i]
                os.makedirs(f"test_instance_mask/{i}", exist_ok=True)
                cv2.imwrite(f"test_instance_mask/{i}/{int(idx.cpu().numpy()):04d}.png", instance_mask_i.cpu().numpy() * 255)


        if results[0]['rgb'] is not None:
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

        fg_rgb = torch.cat([fg_rgb_pred], dim=0).cpu().numpy()
        fg_rgb = (fg_rgb * 255).astype(np.uint8)

        normal = (normal * 255).astype(np.uint8)

        cv2.imwrite(f"test_mask/{id}/{int(idx.cpu().numpy()):04d}.png", pred_mask.cpu().numpy() * 255)
        cv2.imwrite(f"test_rendering/{id}/{int(idx.cpu().numpy()):04d}.png", rgb[:, :, ::-1])
        cv2.imwrite(f"test_normal/{id}/{int(idx.cpu().numpy()):04d}.png", normal[:, :, ::-1])
        cv2.imwrite(f"test_fg_rendering/{id}/{int(idx.cpu().numpy()):04d}.png", fg_rgb[:, :, ::-1])

    def test_step(self, batch, *args, **kwargs):
        # outputs = []
        self.model.eval()
        self.test_step_each_person(batch, id=-1)
        if self.num_person > 1:
            for i in range(self.num_person):
                self.test_step_each_person(batch, id=i)
