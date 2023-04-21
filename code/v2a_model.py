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
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.opt.model.sched_milestones, gamma=self.opt.model.sched_factor)
        return [self.optimizer], [self.scheduler]

    def training_step(self, batch):
        inputs, targets = batch

        batch_idx = inputs["idx"]
        
        device = inputs["smpl_params"].device

        if self.opt_smpl:
            # body_model_params = self.body_model_params(batch_idx)
            body_params_list = [self.body_model_list[i](batch_idx) for i in range(self.num_person)]
            inputs['smpl_trans'] = torch.stack([body_model_params['transl'] for body_model_params in body_params_list], dim=1)
            inputs['smpl_shape'] = torch.stack([body_model_params['betas'] for body_model_params in body_params_list], dim=1)
            global_orient = torch.stack([body_model_params['global_orient'] for body_model_params in body_params_list], dim=1)
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

    def training_epoch_end(self, outputs) -> None:        
        # Canonical mesh update every 20 epochs
        if self.current_epoch != 0 and self.current_epoch % 20 == 0:
            for person_id, smpl_server in enumerate(self.model.smpl_server_list):
                cond = {'smpl': torch.zeros(1, 69).float().cuda()}
                mesh_canonical = generate_mesh(lambda x: self.query_oc(x, cond, person_id=person_id), smpl_server.verts_c[0], point_batch=10000, res_up=2)
                self.model.mesh_v_cano_list[person_id] = torch.tensor(mesh_canonical.vertices[None], device = self.model.mesh_v_cano_list[person_id].device).float()
                self.model.mesh_f_cano_list[person_id] = torch.tensor(mesh_canonical.faces.astype(np.int64), device=self.model.mesh_v_cano_list[person_id].device)
                self.model.mesh_face_vertices_list[person_id] = index_vertices_by_faces(self.model.mesh_v_cano_list[person_id], self.model.mesh_f_cano_list[person_id])
        return super().training_epoch_end(outputs)

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
                cond = {'smpl': inputs["smpl_pose"][:, person_id, 3:] / np.pi}
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
