"""
fit smplh to scans

crated by Xianghui, 12, January 2022

the code is tested
"""

import sys, os
from tracemalloc import start
from turtle import pos
sys.path.append(os.getcwd())
import json
import glob
import lzma
import pickle
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from os.path import exists
from pytorch3d.loss import point_mesh_face_distance
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from fitter.base_fitter import BaseFitter
from lib.body_objectives import batch_get_pose_obj, batch_3djoints_loss
from lib.smpl.priors.th_smpl_prior import get_prior
from lib.smpl.priors.th_hand_prior import HandPrior
from lib.smpl.wrapper_pytorch import SMPLPyTorchWrapperBatchSplitParams, SMPLPyTorchWrapperBatch

from lib.point_mesh_face_distacne import point2face_distance, face2point_distance

import kaolin
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import index_vertices_by_faces, check_sign


from robust_loss_pytorch import lossfun
import trimesh

torch.manual_seed(0)

class SMPLHFitter(BaseFitter):

    def __init__(self, model_root, device='cuda:0', save_name='smpl', debug=False, hands=False):
        super().__init__(model_root, device, save_name, debug, hands)
        self.targets = None
        self.joints = None
        self.smpl = None

    def fit(self, scans, pose_file, gender='male', save_path=None):
        # Batch size
        batch_sz = len(scans)
        
        # Load scans and center them. Once smpl is registered, move it accordingly.
        # Do not forget to change the location of 3D joints/ landmarks accordingly.

        # Set optimization hyper parameters
        iterations, pose_iterations, steps_per_iter, pose_steps_per_iter = 5, 5, 30, 30

        # th_pose_3d = None
        # if pose_file is not None:
        #     th_pose_3d_init = torch.from_numpy(np.load(pose_file)).float().to(self.device)

        th_pose_3d = torch.from_numpy(np.load(pose_file)).unsqueeze(0).float().to(self.device)  # Shape (1, num_frames, 25, 4)
        
        for i in range(0, batch_sz):
            if i == 0:  
                smpl = self.init_smpl(1, gender) # gender actually not used
            else:
                # Set hand, (foot, ankle z) pose to zero
                pose = smpl.pose.detach()
                pose[:,-6:]= 0.0  
                # pose[:,23]= 0.0  
                # pose[:,26]= 0.0 
                # pose[:, 30:36] = 0.0
                smpl = SMPLPyTorchWrapperBatch(self.model_root, 1, smpl.betas.detach(), pose, smpl.trans.detach(),
                                                num_betas=10, device=self.device,
                                                gender=gender, hands=self.hands).to(self.device)
            th_pose_3d_i = th_pose_3d[:,i]

            th_scan_meshes = self.load_scans([scans[i]])

            # Optimize only the pose of first frame
            if i == 0: 
                pose_iterations_init = 5
                self.optimize_pose_only(th_scan_meshes, smpl, pose_iterations_init, pose_steps_per_iter, th_pose_3d_i)


            # Optimize pose and shape
            self.optimize_pose_shape(th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d_i)

            # Refine the shape paramter
            self.refine_pose_shape(th_scan_meshes, smpl, th_pose_3d_i, shape_iters = 3, pose_iters = 3, steps_per_iter=30)

            smpl_path = os.path.join(save_path)
            if not exists(smpl_path):
                os.makedirs(smpl_path)
            poses, betas, trans =  self.save_outputs(smpl_path, [scans[i]], smpl, th_scan_meshes, save_name='smplh' if self.hands else 'smpl')

    def optimize_pose_shape(self, th_scan_meshes, smpl, iterations, steps_per_iter, th_pose_3d=None):
        # Optimizer
        # optimizer = torch.optim.Adam([smpl.trans, smpl.pose], 0.02, betas=(0.9, 0.999))
        optimizer = torch.optim.Adam([smpl.trans, smpl.betas, smpl.pose], 0.02, betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        for it in range(iterations):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Optimizing SMPL')

            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_pose_shape(th_scan_meshes, smpl, th_pose_3d)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                # if self.debug:
                #     self.viz_fitting(smpl, th_scan_meshes)

        print('** Optimised smpl pose and shape **')

    def forward_pose_shape(self, th_scan_meshes, smpl, th_pose_3d=None):
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender)

        # forward
        verts, _, _, _ = smpl()
        th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))
    
        # losses

        loss = dict()
       
        loss['s2m'] = point_mesh_face_distance(th_smpl_meshes, Pointclouds(points=th_scan_meshes.verts_list()))
        loss['m2s'] = point_mesh_face_distance(th_scan_meshes, Pointclouds(points=th_smpl_meshes.verts_list()))
         
        loss['betas'] = torch.mean(smpl.betas ** 2)
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        if self.hands:
            hand_prior = HandPrior(self.model_root, type='grab')
            loss['hand'] = torch.mean(hand_prior(smpl.pose)) # add hand prior if smplh is used
        if th_pose_3d is not None:
            # 3D joints loss
            J, face, hands = smpl.get_landmarks()
            joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
            j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
            loss['pose_obj'] = j3d_loss
            # loss['pose_obj'] = batch_get_pose_obj(th_pose_3d, smpl).mean()
        return loss

    def refine_pose_shape(self, th_scan_meshes, smpl, th_pose_3d_i, shape_iters, pose_iters, steps_per_iter):

        # Get loss_weights
        weight_dict = self.get_loss_weights()

        # Shape refinement
        optimizer = torch.optim.Adam([smpl.betas], 0.02, betas=(0.9, 0.999))

        for it in range(shape_iters):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Refine SMPL shape')

            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_shape(th_scan_meshes, smpl)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                # if self.debug:
                #     self.viz_fitting(smpl, th_scan_meshes)

        # Pose refinement
        optimizer = torch.optim.Adam([smpl.pose], 0.001, betas=(0.9, 0.999))

        for it in range(pose_iters):
            loop = tqdm(range(steps_per_iter))
            loop.set_description('Refine SMPL pose')

            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_pose(th_scan_meshes, smpl, th_pose_3d_i)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it)
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

        print('** Refined smpl shape and pose **')

    def forward_shape(self, th_scan_meshes, smpl):
        # forward
        verts, _, _, _ = smpl()
        # th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))
        samples = sample_points_from_meshes(th_scan_meshes,  num_samples=50000)
    
        # losses
        loss = dict()

        mask = kaolin.ops.mesh.check_sign(verts, smpl.faces, samples) # check if sampled point inside smpl model
        # # loss["out"] = mask.float().mean()

        distances, _ = kaolin.metrics.pointcloud.sided_distance(samples, verts)
        icp_loss = lossfun(distances.sqrt(), alpha=torch.Tensor([-2.]).to(distances.device), scale=torch.Tensor([0.05]).to(distances.device))
        icp_loss[mask] *= 10.0
     
        loss['icp'] = icp_loss.mean()
        loss['betas'] = torch.mean(smpl.betas ** 2)

        return loss    

    def forward_pose(self, th_scan_meshes, smpl, th_pose_3d_i):
        # forward
        verts, _, _, _ = smpl()
        # th_smpl_meshes = Meshes(verts=verts, faces=torch.stack([smpl.faces] * len(verts), dim=0))
        samples = sample_points_from_meshes(th_scan_meshes,  num_samples=50000)
    
        # losses
        loss = dict()

        mask = kaolin.ops.mesh.check_sign(verts, smpl.faces, samples) # check if sampled point inside smpl model

        distances, _ = kaolin.metrics.pointcloud.sided_distance(samples, verts)
        icp_loss = lossfun(distances.sqrt(), alpha=torch.Tensor([-2.]).to(distances.device), scale=torch.Tensor([0.05]).to(distances.device))
        icp_loss[mask] *= 10.0

        loss['icp'] = icp_loss.mean()

        if th_pose_3d_i is not None:
            # 3D joints loss
            J, face, hands = smpl.get_landmarks()
            joints = self.compose_smpl_joints(J, face, hands, th_pose_3d_i)
            j3d_loss = batch_3djoints_loss(th_pose_3d_i, joints)
            loss['pose_obj'] = j3d_loss

        return loss    

    def compose_smpl_joints(self, J, face, hands, th_pose_3d):
        if th_pose_3d.shape[1] == 25:
            joints = J
        else:
            joints = torch.cat([J, face, hands], 1)
        return joints

    def optimize_pose_only(self, th_scan_meshes, smpl, iterations,
                           steps_per_iter, th_pose_3d, prior_weight=None):
        # split_smpl = SMPLHPyTorchWrapperBatchSplitParams.from_smplh(smpl).to(self.device)
        split_smpl = SMPLPyTorchWrapperBatchSplitParams.from_smpl(smpl, fixed_feet=False).to(self.device)
        optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose], 0.05,
                                     betas=(0.9, 0.999))

        # Get loss_weights
        weight_dict = self.get_loss_weights()
        
        iter_for_global = 10
        for it in range(iter_for_global + iterations):
            loop = tqdm(range(steps_per_iter))
            if it < iter_for_global:
                # Optimize global orientation
                print('Optimizing SMPL global orientation')
                loop.set_description('Optimizing SMPL global orientation')
            elif it == iter_for_global:
                # Now optimize full SMPL pose
                print('Optimizing SMPL pose only')
                loop.set_description('Optimizing SMPL pose only')
                optimizer = torch.optim.Adam([split_smpl.trans, split_smpl.top_betas, split_smpl.global_pose,
                                              split_smpl.body_pose], 0.02, betas=(0.9, 0.999))
            else:
                loop.set_description('Optimizing SMPL pose only')

            for i in loop:
                optimizer.zero_grad()
                # Get losses for a forward pass
                loss_dict = self.forward_step_pose_only(split_smpl, th_pose_3d, prior_weight)
                # Get total loss for backward pass
                tot_loss = self.backward_step(loss_dict, weight_dict, it/2)
                tot_loss.backward()
                optimizer.step()

                split_smpl.betas = torch.cat([split_smpl.top_betas.clone(), split_smpl.other_betas.clone()], axis=1)
                split_smpl.pose = torch.cat([split_smpl.global_pose.clone(), split_smpl.body_pose.clone(), split_smpl.hand_pose.clone()], axis=1)

                l_str = 'Iter: {}'.format(i)
                for k in loss_dict:
                    l_str += ', {}: {:0.4f}'.format(k, weight_dict[k](loss_dict[k], it).mean().item())
                    loop.set_description(l_str)

                # if self.debug:
                #     self.viz_fitting(split_smpl, th_scan_meshes)

        self.copy_smpl_params(smpl, split_smpl)
        print('** Optimised smpl pose **')

    def copy_smpl_params(self, smpl, split_smpl):
        # Put back pose, shape and trans into original smpl
        smpl.pose.data = split_smpl.pose.data
        smpl.betas.data = split_smpl.betas.data
        smpl.trans.data = split_smpl.trans.data

    def forward_step_pose_only(self, smpl, th_pose_3d, prior_weight):
        """
        Performs a forward step, given smpl and scan meshes.
        Then computes the losses.
        currently no prior weight implemented for smplh
        """
        # Get pose prior
        prior = get_prior(self.model_root, smpl.gender)
        
        # losses
        loss = dict()
        loss['pose_pr'] = torch.mean(prior(smpl.pose))
        # 3D joints loss
        J, face, hands = smpl.get_landmarks()
        joints = self.compose_smpl_joints(J, face, hands, th_pose_3d)
        j3d_loss = batch_3djoints_loss(th_pose_3d, joints)
        loss['pose_obj'] = j3d_loss
        return loss

    def get_loss_weights(self):
        """Set loss weights"""
        loss_weight = {'s2m': lambda cst, it: 20. ** 2 * cst * (1 + it),   
                       'm2s': lambda cst, it: 20. ** 2 * cst / (1 + it),
                       'betas': lambda cst, it: 10. ** 0 * cst / (1 + it),
                       'offsets': lambda cst, it: 10. ** -1 * cst / (1 + it),
                       'pose_pr': lambda cst, it: 10. ** -2 * cst / (1 + it),
                       'hand': lambda cst, it: 10. ** -5 * cst / (1 + it),
                       'lap': lambda cst, it: cst / (1 + it),
                       'pose_obj': lambda cst, it: 10. ** 2 * cst / (1 + it),
                       'icp': lambda cst, it: 10. ** 1 * cst / (1 + it),
                       }
        return loss_weight

