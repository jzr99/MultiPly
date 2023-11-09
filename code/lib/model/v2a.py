from .networks import ImplicitNet, RenderingNet
from .density import LaplaceDensity, AbsDensity
from .ray_sampler import ErrorBoundSampler
from .deformer import SMPLDeformer
from .smpl import SMPLServer

from .sampler import PointInSpace, PointOnBones

from ..utils import rend_util

import numpy as np
import trimesh
import torch
import torch.nn as nn
from torch.autograd import grad
import hydra
import kaolin
from kaolin.ops.mesh import index_vertices_by_faces
import json
from pytorch3d import ops
class V2A(nn.Module):
    def __init__(self, opt, betas_path):
        super().__init__()

        betas = np.load(betas_path)
        # default: use_depth_order_loss = True
        try:
            self.smpl_surface_weight = opt.loss.smpl_surface_weight
            print("self.smpl_surface_weight ", self.smpl_surface_weight)
        except:
            self.smpl_surface_weight = 0
            print("self.smpl_surface_weight ", self.smpl_surface_weight)

        try:
            self.zero_pose_weight = opt.loss.zero_pose_weight
            print("self.zero_pose_weight ", self.zero_pose_weight)
        except:
            self.zero_pose_weight = 0
            print("self.zero_pose_weight ", self.zero_pose_weight)
        try:
            self.use_depth_order_loss = opt.use_depth_order_loss
            print("self.use_depth_order_loss ", self.use_depth_order_loss)
        except:
            self.use_depth_order_loss = True
            print("self.use_depth_order_loss ", self.use_depth_order_loss)
        # default: use_person_encoder = False
        try:
            self.use_person_encoder = opt.use_person_encoder
            print("self.use_person_encoder ", self.use_person_encoder)
        except:
            self.use_person_encoder = False
            print("self.use_person_encoder ", self.use_person_encoder)
        # human id encoder
        if self.use_person_encoder:
            assert len(betas.shape) == 2
            self.person_latent_encoder = nn.Embedding(betas.shape[0], 64)
            # self.triplane_person_encoder = TriPlane(number_person=betas.shape[0], features=64)

        self.foreground_implicit_network_list = nn.ModuleList()
        self.foreground_rendering_network_list = nn.ModuleList()
        if len(betas.shape) == 2:
            if self.use_person_encoder:
                # we use shared network for all people
                implicit_model = ImplicitNet(opt.implicit_network, betas=betas)
                if opt.rendering_network.mode == 'pose_no_view':
                    print('use shared network only for shape network, separate for render network')
                    for i in range(betas.shape[0]):
                        self.foreground_implicit_network_list.append(implicit_model)
                        self.foreground_rendering_network_list.append(RenderingNet(opt.rendering_network))
                else:
                    print('use shared network for all people')
                    render_model = RenderingNet(opt.rendering_network, triplane=implicit_model.triplane)
                    for i in range(betas.shape[0]):
                        self.foreground_implicit_network_list.append(implicit_model)
                        self.foreground_rendering_network_list.append(render_model)
            else:
                for i in range(betas.shape[0]):
                    self.foreground_implicit_network_list.append(ImplicitNet(opt.implicit_network))
                    self.foreground_rendering_network_list.append(RenderingNet(opt.rendering_network))
        else:
            self.foreground_implicit_network_list.append(ImplicitNet(opt.implicit_network))
            self.foreground_rendering_network_list.append(RenderingNet(opt.rendering_network))
        # # Foreground networks 1
        # self.implicit_network_1 = ImplicitNet(opt.implicit_network)
        # self.rendering_network_1 = RenderingNet(opt.rendering_network)
        #
        # # Foreground networks 2
        # self.implicit_network_2 = ImplicitNet(opt.implicit_network)
        # self.rendering_network_2 = RenderingNet(opt.rendering_network)

        # Background networks
        self.with_bkgd = opt.with_bkgd
        self.bg_implicit_network = ImplicitNet(opt.bg_implicit_network)
        self.bg_rendering_network = RenderingNet(opt.bg_rendering_network)

        # Frame latent encoder
        self.frame_latent_encoder = nn.Embedding(opt.num_training_frames, opt.dim_frame_encoding)
        self.sampler = PointInSpace()

        self.use_smpl_deformer = opt.use_smpl_deformer
        # self.gender = 'male'
        self.gender_list = np.load(betas_path[:-14] + "gender.npy")
        if self.use_smpl_deformer:
            self.deformer_list = torch.nn.ModuleList()
            # self.deformer_list = []
            if len(betas.shape) == 2:
                for i in range(betas.shape[0]):
                    deformer = SMPLDeformer(betas=betas[i], gender=self.gender_list[i])
                    self.deformer_list.append(deformer)
            else:
                deformer = SMPLDeformer(betas=betas, gender='male')
                self.deformer_list.append(deformer)


                # pre-defined bounding sphere
        self.sdf_bounding_sphere = 3.0
        
        # threshold for the out-surface points
        self.threshold = 0.05
        self.density = LaplaceDensity(**opt.density)
        self.bg_density = AbsDensity()

        self.ray_sampler = ErrorBoundSampler(self.sdf_bounding_sphere, inverse_sphere_bg=True, **opt.ray_sampler)
        self.smpl_server_list = torch.nn.ModuleList()
        if len(betas.shape) == 2:
            for i in range(betas.shape[0]):
                smpl_server = SMPLServer(gender=self.gender_list[i], betas=betas[i])
                self.smpl_server_list.append(smpl_server)
        else:
            smpl_server = SMPLServer(gender='male', betas=betas)
            self.smpl_server_list.append(smpl_server)
        # self.smpl_server = SMPLServer(gender=self.gender, betas=betas)
        # self.sampler_bone = PointOnBones(self.smpl_server.bone_ids)

        if opt.smpl_init:
            if self.use_person_encoder:
                smpl_model_state = torch.load(hydra.utils.to_absolute_path('./outputs/smpl_init_%s_256_id.pth' % 'male'))
            else:
                smpl_model_state = torch.load(hydra.utils.to_absolute_path('./outputs/smpl_init_%s_256.pth' % 'male'))
            for implicit_network in self.foreground_implicit_network_list:
                implicit_network.load_state_dict(smpl_model_state["model_state_dict"], strict=False)
            if not self.use_smpl_deformer:
                self.deformer.load_state_dict(smpl_model_state["deformer_state_dict"])

        if self.smpl_surface_weight > 0:
            self.smpl_vertex_part = json.load(open(hydra.utils.to_absolute_path('./outputs/smpl_vert_segmentation.json')))

        # self.smpl_v_cano = self.smpl_server.verts_c
        # self.smpl_f_cano = torch.tensor(self.smpl_server.smpl.faces.astype(np.int64), device=self.smpl_v_cano.device)
        # DEBUG
        self.mesh_v_cano_list = []
        self.mesh_f_cano_list = []
        self.mesh_face_vertices_list = []
        for smpl_server in self.smpl_server_list:
            self.mesh_v_cano_list.append(smpl_server.verts_c)
            self.mesh_f_cano_list.append(torch.tensor(smpl_server.smpl.faces.astype(np.int64), device=smpl_server.verts_c.device))
            self.mesh_face_vertices_list.append(index_vertices_by_faces(self.mesh_v_cano_list[-1], self.mesh_f_cano_list[-1]))

        self.sam_0_mask = None
        self.sam_1_mask = None
        # if True:
        #     self.sam_0_mask = np.load('/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/courtyard_shakeHands_00_loop/V2A_mask/0_sam_opt.npy')
        #     self.sam_1_mask = np.load(
        #         '/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/courtyard_shakeHands_00_loop/V2A_mask/1_sam_opt.npy')

    def sdf_func_with_smpl_deformer(self, x, cond, smpl_tfs, smpl_verts, person_id):
        if hasattr(self, "deformer_list"):
            x_c, outlier_mask = self.deformer_list[person_id].forward(x, smpl_tfs, return_weights=False, inverse=True, smpl_verts=smpl_verts)
            output = self.foreground_implicit_network_list[person_id](x_c, cond, person_id=person_id)[0]
            sdf = output[:, 0:1]
            if not self.training:
                sdf[outlier_mask] = 4. # set a large SDF value for outlier points
            ''' Clamping the SDF with the scene bounding sphere, so that all rays are eventually occluded '''
            if not self.with_bkgd:
                if self.sdf_bounding_sphere > 0.0:
                    sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
                    sdf = torch.minimum(sdf, sphere_sdf)
            feature = output[:, 1:]

        return sdf, x_c, feature
    
    def check_off_in_surface_points_cano_mesh(self, x_cano, N_samples, person_id, threshold=0.05):

        distance, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(x_cano.unsqueeze(0).contiguous(), self.mesh_face_vertices_list[person_id])

        distance = torch.sqrt(distance) # kaolin outputs squared distance
        sign = kaolin.ops.mesh.check_sign(self.mesh_v_cano_list[person_id], self.mesh_f_cano_list[person_id], x_cano.unsqueeze(0)).float()
        sign = 1 - 2 * sign
        signed_distance = sign * distance
        batch_size = x_cano.shape[0] // N_samples
        signed_distance = signed_distance.reshape(batch_size, N_samples, 1)

        minimum = torch.min(signed_distance, 1)[0]
        index_off_surface = (minimum > threshold).squeeze(1)
        index_in_surface = (minimum <= 0.).squeeze(1)
        return index_off_surface, index_in_surface

    def query_oc(self, x, cond, person_id):

        x = x.reshape(-1, 3)
        mnfld_pred = self.foreground_implicit_network_list[person_id](x, cond, person_id=person_id)[:,:,0].reshape(-1,1)
        return {'occ':mnfld_pred}
    def forward(self, input, id=-1, cond_zero_shit=False):
        # Parse model input
        torch.set_grad_enabled(True)
        # if self.sam_0_mask is not None:
        #     sam_mask = torch.from_numpy(np.stack([self.sam_0_mask[input['idx']], self.sam_1_mask[input['idx']]], axis=0)).float().to(input["smpl_pose"].device)
        # else:
        #     sam_mask = None
        intrinsics = input["intrinsics"]
        pose = input["pose"]
        uv = input["uv"]

        smpl_params = input['smpl_params']
        # smpl_thetas = smpl_params[:, 4:76]
        smpl_pose = input["smpl_pose"]
        # scale, transl, _, betas = torch.split(smpl_params, [1, 3, 72, 10], dim=1)
        scale = smpl_params[:,:, 0]
        smpl_shape = input["smpl_shape"]
        smpl_trans = input["smpl_trans"]
        # (B, P, 3)
        num_person = smpl_trans.shape[1]
        smpl_tfs_list = []
        smpl_output_list = []
        raymeshintersector_list = []
        for i in range(num_person):
            smpl_output = self.smpl_server_list[i](scale[:,i], smpl_trans[:,i], smpl_pose[:,i], smpl_shape[:,i])
            smpl_output_list.append(smpl_output)
            smpl_tfs_list.append(smpl_output['smpl_tfs'])
            # load into trimesh
            smpl_mesh = trimesh.Trimesh(vertices=smpl_output['smpl_verts'][0].detach().cpu().numpy(),
                                        faces=self.smpl_server_list[i].faces, process=False)
            if self.training and self.use_depth_order_loss:
                raymeshintersector_list.append(trimesh.ray.ray_triangle.RayMeshIntersector(smpl_mesh))
            # TODO need to comfirm that camera ray is aligned with scaled smpl



        # smpl_params = smpl_params.detach()
        # smpl_mesh = trimesh.Trimesh(vertices=smpl_output['smpl_verts'][0].detach().cpu().numpy(), faces=self.smpl_server.faces, process=False)
        # import pdb;pdb.set_trace()

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, num_pixels, _ = ray_dirs.shape

        cam_loc = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        fg_rgb_list = []
        normal_values_list = []
        sdf_output_list = []
        z_vals_list = []
        person_id_list = []
        z_max_list = []
        index_off_surface_list = []
        index_in_surface_list = []
        grad_theta_list = []
        interpenetration_loss = torch.zeros(1,device=smpl_pose.device)
        temporal_loss = torch.zeros(1, device=smpl_pose.device)
        smpl_surface_loss = torch.zeros(1, device=smpl_pose.device)
        zero_pose_loss = torch.zeros(1, device=smpl_pose.device)
        # if input['current_epoch'] < 500 and input['idx'] > 85 and input['idx'] < 125:
        if self.training and input['current_epoch'] > 250:
            temporal_loss = torch.mean(torch.square(input["smpl_pose_last"] - input["smpl_pose"]))
        if id == -1:
            person_list = range(num_person)
        else:
            person_list = [id]
        # # use bounding box to determine the hitted person
        # for person_id in person_list:
        #     bounding_box_max = smpl_output_list[person_id]['smpl_verts'][0].max(dim=0)
        #     bounding_box_min = smpl_output_list[person_id]['smpl_verts'][0].min(dim=0)
        #     # determine camera ray hit which person
        #     index_ray = (cam_loc[:, 0] > bounding_box_min[0]) & (cam_loc[:, 0] < bounding_box_max[0]) & \
        #                 (cam_loc[:, 1] > bounding_box_min[1]) & (cam_loc[:, 1] < bounding_box_max[1]) & \
        #                 (cam_loc[:, 2] > bounding_box_min[2]) & (cam_loc[:, 2] < bounding_box_max[2])
        #     # index_ray_list.append(index_ray)
        #     # person_id_list.append(person_id)

        hitted_face_idx_list = []
        index_triangle_list = []
        index_ray_list = []
        locations_list = []
        for person_id in person_list:
            cond_pose = smpl_pose[:, person_id, 3:] / np.pi
            if self.training:
                if input['current_epoch'] < 20 or input['current_epoch'] % 20 == 0 or cond_zero_shit:
                    cond_pose = smpl_pose[:, person_id, 3:] * 0.
                    # cond = {'smpl': smpl_pose[:, person_id, 3:] * 0.}
                # if input['current_epoch'] < 500:
                #     cond = {'smpl': smpl_pose[:, person_id, 3:] * 0.}

            # cond = {'smpl': smpl_pose[:, person_id, 3:] / np.pi}
            if self.use_person_encoder:
                # import pdb;pdb.set_trace()
                person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(smpl_pose.device)
                person_encoding = self.person_latent_encoder(person_id_tensor)
                person_encoding = person_encoding.repeat(smpl_pose.shape[0], 1)
                cond_pose_id = torch.cat([cond_pose, person_encoding], dim=1)
                cond = {'smpl_id': cond_pose_id}
            else:
                cond = {'smpl': cond_pose}
            # if True:
            #     cond = {'smpl': input["smpl_pose_cond_24"][:, person_id, 3:] / np.pi}
            z_vals, _ = self.ray_sampler.get_z_vals(ray_dirs, cam_loc, self, cond, smpl_tfs_list[person_id], eval_mode=True, smpl_verts=smpl_output_list[person_id]['smpl_verts'], person_id=person_id)

            z_vals, z_vals_bg = z_vals
            z_max = z_vals[:,-1]
            z_vals = z_vals[:,:-1]
            N_samples = z_vals.shape[1]

            points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
            points_flat = points.reshape(-1, 3)
            # import pdb;pdb.set_trace()
            # depth order loss related
            if self.training and self.use_depth_order_loss:
                face_hitted_list = raymeshintersector_list[person_id].intersects_first(cam_loc.cpu(), ray_dirs.cpu())
                hitted_face_idx_list.append(face_hitted_list)
                index_triangle, index_ray, locations = raymeshintersector_list[person_id].intersects_id(cam_loc.cpu(), ray_dirs.cpu(), multiple_hits=False, return_locations=True)
                index_triangle_list.append(index_triangle)
                index_ray_list.append(index_ray)
                locations_list.append(locations)


            dirs = ray_dirs.unsqueeze(1).repeat(1,N_samples,1)
            sdf_output, canonical_points, feature_vectors = self.sdf_func_with_smpl_deformer(points_flat, cond, smpl_tfs_list[person_id], smpl_verts=smpl_output_list[person_id]['smpl_verts'], person_id=person_id)

            sdf_output = sdf_output.unsqueeze(1)

            if self.training:
                if input['current_epoch'] < 250:
                    index_off_surface_person, index_in_surface_person = self.check_off_in_surface_points_cano_mesh(canonical_points, N_samples, person_id=person_id, threshold=self.threshold)
                    index_in_surface_list.append(index_in_surface_person)
                    index_off_surface_list.append(index_off_surface_person)
                canonical_points = canonical_points.reshape(num_pixels, N_samples, 3)

                canonical_points = canonical_points.reshape(-1, 3)

                # sample canonical SMPL surface pnts for the eikonal loss
                smpl_verts_c = self.smpl_server_list[person_id].verts_c.repeat(batch_size, 1,1)

                indices = torch.randperm(smpl_verts_c.shape[1])[:num_pixels].cuda()
                verts_c = torch.index_select(smpl_verts_c, 1, indices)
                sample = self.sampler.get_points(verts_c, global_ratio=0.)

                sample.requires_grad_()
                local_pred = self.foreground_implicit_network_list[person_id](sample, cond, person_id=person_id)[..., 0:1]
                grad_theta_person = gradient(sample, local_pred)
                grad_theta_list.append(grad_theta_person)

                differentiable_points = canonical_points

                # sample point form deformed SMPL
                if self.smpl_surface_weight > 0:
                    assert smpl_output_list[person_id]['smpl_verts'].shape[1] == 6890
                    idx_weight = torch.ones(6890)
                    exclude_idx = self.smpl_vertex_part['head'] + self.smpl_vertex_part['rightHand'] + self.smpl_vertex_part['leftHand'] + self.smpl_vertex_part['rightFoot'] + self.smpl_vertex_part['leftFoot'] + self.smpl_vertex_part['leftHandIndex1'] + self.smpl_vertex_part['rightHandIndex1']
                    idx_weight[exclude_idx] = 0
                    idx_weight = idx_weight.cuda()
                    idx = idx_weight.multinomial(num_samples=num_pixels, replacement=True)
                    # idx = torch.randperm(smpl_output_list[person_id]['smpl_verts'].shape[1])[:num_pixels].cuda()
                    sample_point = torch.index_select(smpl_output_list[person_id]['smpl_verts'], dim=1, index=idx)
                    x_c, outlier_mask = self.deformer_list[person_id].forward(sample_point.reshape(-1, 3),
                                                                               smpl_tfs_list[person_id],
                                                                               return_weights=False,
                                                                               inverse=True,
                                                                               smpl_verts=smpl_output_list[person_id][
                                                                                   'smpl_verts'])
                    output = self.foreground_implicit_network_list[person_id](x_c, cond, person_id=person_id)[0]
                    sdf = output[:, 0:1]
                    sdf = sdf.reshape(-1)
                    threshold_smpl_sdf = 0.02
                    if (sdf > threshold_smpl_sdf).any():
                        # TODO: Eikonal loss should be sampled also from the global
                        unplausible_sdf = sdf[sdf > threshold_smpl_sdf]
                        sdf_zero = torch.ones_like(unplausible_sdf, device=sdf.device) * threshold_smpl_sdf
                        smpl_surface_loss += torch.nn.functional.l1_loss(unplausible_sdf, sdf_zero, reduction='mean')

                sample_pixel = 512
                if self.zero_pose_weight > 0:
                    for p, canonical_average_vertex in enumerate(self.mesh_v_cano_list):
                        idx = torch.randperm(canonical_average_vertex.shape[1])[:sample_pixel].cuda()
                        sample_point = torch.index_select(canonical_average_vertex, dim=1, index=idx)
                        output_pred = self.foreground_implicit_network_list[p](sample_point.reshape(-1,3), cond, person_id=p)[0]
                        sdf_pred = output_pred[:, 0:1]


                        cond_zero_pose = smpl_pose[:, person_id, 3:] * 0.
                        if self.use_person_encoder:
                            # import pdb;pdb.set_trace()
                            person_id_tensor = torch.from_numpy(np.array([person_id])).long().to(smpl_pose.device)
                            person_encoding = self.person_latent_encoder(person_id_tensor)
                            person_encoding = person_encoding.repeat(smpl_pose.shape[0], 1)
                            cond_zero_pose_id = torch.cat([cond_zero_pose, person_encoding], dim=1)
                            cond_zero = {'smpl_id': cond_zero_pose_id}
                        else:
                            cond_zero = {'smpl': cond_zero_pose}

                        output_zero = self.foreground_implicit_network_list[p](sample_point.reshape(-1,3), cond_zero, person_id=p)[0]
                        sdf_zero = output_zero[:, 0:1]
                        zero_pose_loss += torch.nn.functional.l1_loss(sdf_pred, sdf_zero, reduction='mean')
                        feature_zero = output_zero[:, 1:]
                        feature_pred = output_pred[:, 1:]
                        zero_pose_loss += torch.nn.functional.l1_loss(feature_pred, feature_zero, reduction='mean')

                # TODO: implement 3 person interpenetration loss
                # if len(person_list) == 2:
                #     # sample point for interpenetration loss
                #     assert smpl_output_list[person_id]['smpl_verts'].shape[1] == 6890
                #     assert len(person_list) == 2
                #     idx = torch.randperm(smpl_output_list[person_id]['smpl_verts'].shape[1])[:num_pixels].cuda()
                #     sample_point = torch.index_select(smpl_output_list[person_id]['smpl_verts'],dim=1,index=idx)
                #     partner_id = 1-person_id
                #
                #     x_c, outlier_mask = self.deformer_list[partner_id].forward(sample_point.reshape(-1,3), smpl_tfs_list[partner_id], return_weights=False,
                #                                                               inverse=True, smpl_verts=smpl_output_list[partner_id]['smpl_verts'])
                #     output = self.foreground_implicit_network_list[partner_id](x_c, cond, person_id=partner_id)[0]
                #     sdf = output[:, 0:1]
                #     sdf = sdf.reshape(-1)
                #     if (sdf < -0.01).any():
                #         penetrate_point = sample_point[:, sdf < -0.01]
                #         # if there is a possibility that it actually make it penetrate? like smpl is outside but implicit is inside?
                #         # TODO try to penelize implicit function
                #         distance_batch, index_batch, neighbor_points = ops.knn_points(penetrate_point, smpl_output_list[partner_id]['smpl_verts'], K=1, return_nn=True)
                #         # (N, P1, K, D) for neighbor_points
                #         neighbor_points = neighbor_points.mean(dim=2)
                #         # filter outlier point
                #         stable_point = (penetrate_point - neighbor_points).norm(dim=-1).reshape(-1) < 0.5
                #         if stable_point.any():
                #             penetrate_point = penetrate_point.reshape(-1, 3)[stable_point]
                #             neighbor_points = neighbor_points.reshape(-1, 3)[stable_point]
                #             # TODO make the neightbor_points move torwards its normal by a little margin
                #             interpenetration_loss = interpenetration_loss + torch.nn.functional.mse_loss(penetrate_point, neighbor_points)

            else:
                differentiable_points = canonical_points.reshape(num_pixels, N_samples, 3).reshape(-1, 3)
                grad_theta = None

            sdf_output = sdf_output.reshape(num_pixels, N_samples, 1).reshape(-1, 1)
            sdf_output_list.append(sdf_output.reshape(num_pixels, N_samples))
            person_label = torch.ones_like(sdf_output.reshape(num_pixels, N_samples), device=sdf_output.device) * person_id
            z_vals = z_vals
            view = -dirs.reshape(-1, 3)

            if differentiable_points.shape[0] > 0:
                # if True:
                #     # cond = {'smpl': smpl_pose[:, person_id, 3:] * 0.}
                #     cond = {'smpl': input["smpl_pose_cond_24"][:, person_id, 3:] / np.pi}
                fg_rgb_flat, others = self.get_rbg_value(points_flat, differentiable_points, view,
                                                         cond, smpl_tfs_list[person_id], feature_vectors=feature_vectors, person_id=person_id, is_training=self.training)
                normal_values = others['normals']

            if 'image_id' in input.keys():
                frame_latent_code = self.frame_latent_encoder(input['image_id'])
            else:
                frame_latent_code = self.frame_latent_encoder(input['idx'])

            fg_rgb = fg_rgb_flat.reshape(-1, N_samples, 3)
            fg_rgb_list.append(fg_rgb)
            normal_values = normal_values.reshape(-1, N_samples, 3)
            normal_values_list.append(normal_values)
            z_max_list.append(z_max)
            z_vals_list.append(z_vals)
            person_id_list.append(person_label)

        fg_rgb_values_each_person_list = []
        t_list = []
        mean_hitted_vertex_list = []
        hitted_mask_idx = []
        # TODO: for nerfacc version, we need to change person-specific index_ray_list to global index_ray_list, and discard using hitted_face_idx_list
        if self.training and self.use_depth_order_loss:
            hitted_face_idx_list = np.stack(hitted_face_idx_list, axis=0)
            hitted_face_mask = hitted_face_idx_list > 0
            # TODO this is wrong, change to sum >= 2 (done)
            # TODO what if smpl is not overlap but implicit is overlap? should query ray intersection with deformed implicit surface
            hitted_face_mask = np.sum(hitted_face_mask, axis=0) >= 2
            # hitted_face_mask = np.prod(hitted_face_mask, axis=0)
            # volume render each person separately
            hitted_mask_idx = np.where(hitted_face_mask)[0]
            hitted_face_idx = hitted_face_idx_list[:, hitted_mask_idx]
            if len(hitted_mask_idx) > 0:
                for person_id in person_list:
                    index_triangle_i, index_ray_i, locations_i = index_triangle_list[person_id], index_ray_list[person_id], locations_list[person_id]
                    # sort index_ray_i
                    sorted_idx = np.argsort(index_ray_i)
                    index_ray_i = index_ray_i[sorted_idx]
                    index_triangle_i = index_triangle_i[sorted_idx]
                    locations_i = locations_i[sorted_idx]
                    # hitted_mask_idx store the ray that intersect with more than 2 person, so we need to filter out the ray that only intersect with one person
                    locations_i_idx = np.nonzero(np.in1d(index_ray_i, hitted_mask_idx))[0]
                    locations_i = locations_i[locations_i_idx]
                    index_triangle_i = index_triangle_i[locations_i_idx]
                    index_ray_i = index_ray_i[locations_i_idx]
                    assert (index_ray_i == hitted_mask_idx).all()
                    assert (hitted_face_idx[person_id] == index_triangle_i).all()
                    # TODO this may be wrong for person > 2, need an additional mask to indicate which person is hitted
                    assert (index_triangle_i != -1).all()
                    # (number_hitted_faces, 3)
                    vertex_index_i = self.smpl_server_list[person_id].faces[index_triangle_i]
                    # import pdb;pdb.set_trace()
                    vertex_index_i = vertex_index_i.reshape(-1)
                    vertex_index_i = torch.from_numpy(vertex_index_i.astype('int64')).cuda()
                    vertex_position = smpl_output_list[person_id]['smpl_verts'][0][vertex_index_i]
                    vertex_position = vertex_position.reshape(-1, 3, 3)
                    mean_hitted_vertex_list.append(vertex_position.mean(dim=1))
                    # TODO check if it is not away from location_i (done error around 5mm)


                    t = (locations_i - cam_loc[index_ray_i].cpu().numpy()) / (ray_dirs[index_ray_i].cpu().numpy() + 1e-12)
                    # TODO insure t is same for dimension -1
                    t = t[:,0]
                    t_list.append(t)
                    # index_triangle_i should be same as hitted_face_idx
                    # TODO still can reduce some computation power by disable when only one person in the scene
                    z_vals_i = z_vals_list[person_id]
                    fg_rgb_i = fg_rgb_list[person_id]
                    z_max_i = z_max_list[person_id]
                    sdf_output_i = sdf_output_list[person_id]
                    # idx = np.where(hitted_face_mask)[0]
                    z_vals_i = z_vals_i[hitted_mask_idx]
                    fg_rgb_i = fg_rgb_i[hitted_mask_idx]
                    z_max_i = z_max_i[hitted_mask_idx]
                    sdf_output_i = sdf_output_i[hitted_mask_idx]
                    # TODO here I didn't consider the sum off the weight
                    weights_i, bg_transmittance_i = self.volume_rendering(z_vals_i, z_max_i, sdf_output_i)
                    fg_rgb_values_i = torch.sum(weights_i.unsqueeze(-1) * fg_rgb_i, 1)
                    fg_rgb_values_each_person_list.append(fg_rgb_values_i)
                mean_hitted_vertex_list = torch.stack(mean_hitted_vertex_list, dim=0)
                # (2, 7)
                fg_rgb_values_each_person_list = torch.stack(fg_rgb_values_each_person_list, dim=0)
                # (2,7,3)
                t_list = np.stack(t_list, axis=0)
                # import pdb; pdb.set_trace()
        # DEBUG z_vals_bg use only last persons
        # DEBUG z_max is the same
        person_id_cat = torch.cat(person_id_list, dim=1)
        z_vals = torch.cat(z_vals_list, dim=1)
        sdf_output = torch.cat(sdf_output_list, dim=1)
        fg_rgb = torch.cat(fg_rgb_list, dim=1)
        normal_values = torch.cat(normal_values_list, dim=1)
        z_vals, sorted_index = torch.sort(z_vals,descending=False,dim=1)

        d1, d2 = sorted_index.shape
        d1_index = torch.arange(d1).unsqueeze(1).repeat((1, d2))
        sdf_output = sdf_output[d1_index, sorted_index]
        fg_rgb = fg_rgb[d1_index, sorted_index]
        normal_values = normal_values[d1_index, sorted_index]
        person_id_cat = person_id_cat[d1_index, sorted_index]
        # sdf_output = sdf_output[sorted_index].reshape(-1,1)
        # fg_rgb = fg_rgb[sorted_index]
        # normal_values = normal_values[sorted_index]
        # import pdb; pdb.set_trace()
        weights, bg_transmittance = self.volume_rendering(z_vals, z_max, sdf_output)
        # import pdb; pdb.set_trace()
        fg_rgb_values = torch.sum(weights.unsqueeze(-1) * fg_rgb, 1)
        acc_person_list = []
        for person_id in person_list:
            number_pixel = weights.shape[0]
            weight_idx = (person_id_cat==person_id).reshape(-1)
            weight_person_i = weights.reshape(-1)[weight_idx]
            weight_person_i = weight_person_i.reshape(number_pixel, -1)
            # weight_person_i = weights[person_id_cat==person_id]
            acc_person_i = torch.sum(weight_person_i, dim=-1)
            acc_person_list.append(acc_person_i)
        # import ipdb;ipdb.set_trace()
        acc_person = torch.stack(acc_person_list, dim=1)
        # Background rendering
        if input['idx'] is not None:
            N_bg_samples = z_vals_bg.shape[1]
            z_vals_bg = torch.flip(z_vals_bg, dims=[-1, ])  # 1--->0

            bg_dirs = ray_dirs.unsqueeze(1).repeat(1,N_bg_samples,1)
            bg_locs = cam_loc.unsqueeze(1).repeat(1,N_bg_samples,1)

            bg_points = self.depth2pts_outside(bg_locs, bg_dirs, z_vals_bg)  # [..., N_samples, 4]
            bg_points_flat = bg_points.reshape(-1, 4)
            bg_dirs_flat = bg_dirs.reshape(-1, 3)
            bg_output = self.bg_implicit_network(bg_points_flat, {'frame': frame_latent_code})[0]
            bg_sdf = bg_output[:, :1]
            bg_feature_vectors = bg_output[:, 1:]
            
            bg_rendering_output = self.bg_rendering_network(None, None, bg_dirs_flat, None, bg_feature_vectors, frame_latent_code)
            if bg_rendering_output.shape[-1] == 4:
                bg_rgb_flat = bg_rendering_output[..., :-1]
                shadow_r = bg_rendering_output[..., -1]
                bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)
                shadow_r = shadow_r.reshape(-1, N_bg_samples, 1)
                bg_rgb = (1 - shadow_r) * bg_rgb
            else:
                bg_rgb_flat = bg_rendering_output
                bg_rgb = bg_rgb_flat.reshape(-1, N_bg_samples, 3)
            bg_weights = self.bg_volume_rendering(z_vals_bg, bg_sdf)
            bg_rgb_values = torch.sum(bg_weights.unsqueeze(-1) * bg_rgb, 1)
        else:
            bg_rgb_values = torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)

        # Composite foreground and background
        bg_rgb_values = bg_transmittance.unsqueeze(-1) * bg_rgb_values
        rgb_values = fg_rgb_values + bg_rgb_values

        normal_values = torch.sum(weights.unsqueeze(-1) * normal_values, 1)

        if self.training:
            if input['current_epoch'] < 250:
                index_off_surface = torch.all(torch.stack(index_off_surface_list, dim=0), dim=0)
                index_in_surface = torch.any(torch.stack(index_in_surface_list, dim=0), dim=0)
            else:
                index_off_surface = None
                index_in_surface = None
            grad_theta = torch.cat(grad_theta_list, dim=1)
            output = {
                # 'sam_mask': sam_mask,
                # 'uv': input["uv"],
                'zero_pose_loss': zero_pose_loss,
                't_list': t_list,
                'fg_rgb_values_each_person_list' : fg_rgb_values_each_person_list,
                'cam_loc': cam_loc,
                # 'smpl_output_list': smpl_output_list,
                'hitted_mask_idx': hitted_mask_idx,
                'mean_hitted_vertex_list': mean_hitted_vertex_list,
                'points': points,
                'rgb_values': rgb_values,
                'normal_values': normal_values,
                'index_outside': input['index_outside'],
                'index_off_surface': index_off_surface,
                'index_in_surface': index_in_surface,
                'acc_map': torch.sum(weights, -1),
                'sdf_output': sdf_output,
                'grad_theta': grad_theta,
                'interpenetration_loss': interpenetration_loss,
                'temporal_loss': temporal_loss,
                'acc_person_list': acc_person, # Number pixel, Number person
                'smpl_surface_loss': smpl_surface_loss,
                'epoch': input['current_epoch'],
            }
            # import pdb;pdb.set_trace()
            if 'sam_mask' in input.keys():
                output.update({"sam_mask": input['sam_mask'].squeeze()})
        else:
            fg_output_rgb = fg_rgb_values + bg_transmittance.unsqueeze(-1) * torch.ones_like(fg_rgb_values, device=fg_rgb_values.device)
            output = {
                'acc_map': torch.sum(weights, -1),
                'acc_person_list': acc_person,
                'rgb_values': rgb_values,
                'fg_rgb_values': fg_output_rgb,
                'normal_values': normal_values,
                'sdf_output': sdf_output,
            }
        return output

    def get_rbg_value(self, x, points, view_dirs, cond, tfs, feature_vectors, person_id, is_training=True):
        pnts_c = points
        others = {}

        _, gradients, feature_vectors = self.forward_gradient(x, pnts_c, cond, tfs, person_id, create_graph=is_training, retain_graph=is_training)
        # ensure the gradient is normalized
        normals = nn.functional.normalize(gradients, dim=-1, eps=1e-6)
        if self.use_person_encoder:
            tri_feat = self.foreground_implicit_network_list[person_id].triplane_feature_list[person_id]
            fg_rendering_output = self.foreground_rendering_network_list[person_id](pnts_c, normals, view_dirs,
                                                                                    cond['smpl_id'][:, :69],
                                                                                    feature_vectors, person_id=person_id, id_latent_code=cond['smpl_id'][:,69:], tri_feat=tri_feat)
        else:
            fg_rendering_output = self.foreground_rendering_network_list[person_id](pnts_c, normals, view_dirs, cond['smpl'],
                                                     feature_vectors, person_id=person_id)
        
        rgb_vals = fg_rendering_output[:, :3]
        others['normals'] = normals
        return rgb_vals, others

    def forward_gradient(self, x, pnts_c, cond, tfs, person_id, create_graph=True, retain_graph=True):
        if pnts_c.shape[0] == 0:
            return pnts_c.detach()
        pnts_c.requires_grad_(True)

        pnts_d = self.deformer_list[person_id].forward_skinning(pnts_c.unsqueeze(0), None, tfs).squeeze(0)
        num_dim = pnts_d.shape[-1]
        grads = []
        for i in range(num_dim):
            d_out = torch.zeros_like(pnts_d, requires_grad=False, device=pnts_d.device)
            d_out[:, i] = 1
            # import ipdb;ipdb.set_trace()
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

        output = self.foreground_implicit_network_list[person_id](pnts_c, cond, person_id=person_id)[0]
        sdf = output[:, :1]

        if not self.with_bkgd:
            if self.sdf_bounding_sphere > 0.0:
                sphere_sdf = self.sphere_scale * (self.sdf_bounding_sphere - x.norm(2,1, keepdim=True))
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

    def volume_rendering(self, z_vals, z_max, sdf):
        density_flat = self.density(sdf)
        density = density_flat.reshape(-1, z_vals.shape[1]) # (batch_size * num_pixels) x N_samples

        # included also the dist from the sphere intersection
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, z_max.unsqueeze(-1) - z_vals[:, -1:]], -1)

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).cuda(), free_energy], dim=-1)  # add 0 for transperancy 1 at t_0
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        fg_transmittance = transmittance[:, :-1]
        weights = alpha * fg_transmittance  # probability of the ray hits something here
        bg_transmittance = transmittance[:, -1]  # factor to be multiplied with the bg volume rendering

        return weights, bg_transmittance

    def bg_volume_rendering(self, z_vals_bg, bg_sdf):
        bg_density_flat = self.bg_density(bg_sdf)
        bg_density = bg_density_flat.reshape(-1, z_vals_bg.shape[1]) # (batch_size * num_pixels) x N_samples

        bg_dists = z_vals_bg[:, :-1] - z_vals_bg[:, 1:]
        bg_dists = torch.cat([bg_dists, torch.tensor([1e10]).cuda().unsqueeze(0).repeat(bg_dists.shape[0], 1)], -1)

        # LOG SPACE
        bg_free_energy = bg_dists * bg_density
        bg_shifted_free_energy = torch.cat([torch.zeros(bg_dists.shape[0], 1).cuda(), bg_free_energy[:, :-1]], dim=-1)  # shift one step
        bg_alpha = 1 - torch.exp(-bg_free_energy)  # probability of it is not empty here
        bg_transmittance = torch.exp(-torch.cumsum(bg_shifted_free_energy, dim=-1))  # probability of everything is empty up to now
        bg_weights = bg_alpha * bg_transmittance # probability of the ray hits something here

        return bg_weights
    
    def depth2pts_outside(self, ray_o, ray_d, depth):

        '''
        ray_o, ray_d: [..., 3]
        depth: [...]; inverse of distance to sphere origin
        '''

        o_dot_d = torch.sum(ray_d * ray_o, dim=-1)
        under_sqrt = o_dot_d ** 2 - ((ray_o ** 2).sum(-1) - self.sdf_bounding_sphere ** 2)
        d_sphere = torch.sqrt(under_sqrt) - o_dot_d
        p_sphere = ray_o + d_sphere.unsqueeze(-1) * ray_d
        p_mid = ray_o - o_dot_d.unsqueeze(-1) * ray_d
        p_mid_norm = torch.norm(p_mid, dim=-1)

        rot_axis = torch.cross(ray_o, p_sphere, dim=-1)
        rot_axis = rot_axis / torch.norm(rot_axis, dim=-1, keepdim=True)
        phi = torch.asin(p_mid_norm / self.sdf_bounding_sphere)
        theta = torch.asin(p_mid_norm * depth)  # depth is inside [0, 1]
        rot_angle = (phi - theta).unsqueeze(-1)  # [..., 1]

        # now rotate p_sphere
        # Rodrigues formula: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        p_sphere_new = p_sphere * torch.cos(rot_angle) + \
                       torch.cross(rot_axis, p_sphere, dim=-1) * torch.sin(rot_angle) + \
                       rot_axis * torch.sum(rot_axis * p_sphere, dim=-1, keepdim=True) * (1. - torch.cos(rot_angle))
        p_sphere_new = p_sphere_new / torch.norm(p_sphere_new, dim=-1, keepdim=True)
        pts = torch.cat((p_sphere_new, depth.unsqueeze(-1)), dim=-1)

        return pts

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