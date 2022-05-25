'''
Takes in smpl parms and initialises a smpl object with optimizable params.
class th_SMPL currently does not take batch dim.
If code works:
    Author: Bharat
else:
    Author: Anonymous
'''
from typing import List
import torch
import torch.nn as nn

from lib.smpl.smplpytorch.smplpytorch.pytorch.smpl_layer import SMPL_Layer
from lib.body_objectives import torch_pose_obj_data
from lib.torch_functions import batch_sparse_dense_matmul
from .const import *


class SMPLPyTorchWrapperBatch(nn.Module):
    def __init__(self, model_root, batch_sz,
                 betas=None, pose=None,
                 trans=None, offsets=None,
                 gender='male', num_betas=300, hands=False,
                 device='cuda:0'):
        super(SMPLPyTorchWrapperBatch, self).__init__()
        self.model_root = model_root
        self.hands = hands # use smpl-h or not
        self.device = device

        if betas is None:
            self.betas = nn.Parameter(torch.zeros(batch_sz, 300))
        else:
            assert betas.ndim == 2
            self.betas = nn.Parameter(betas)
        pose_param_num = SMPLH_POSE_PRAMS_NUM if hands else SMPL_POSE_PRAMS_NUM
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(batch_sz, pose_param_num))
        else:
            assert pose.ndim == 2, f'the given pose shape {pose.shape} is not a batch pose'
            assert pose.shape[1] == pose_param_num, f'given pose param shape {pose.shape} ' \
                                                    f'does not match the model selected: hands={hands}'
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890, 3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        # self.faces = faces
        self.gender = gender

        # pytorch smpl
        if type(gender) is list:
            self.smpl = []
            for g in gender:
                self.smpl.append(SMPL_Layer(center_idx=0, gender=g, num_betas=num_betas,
                                model_root=str(model_root), hands=hands).to(self.device))
            self.faces = self.smpl[0].th_faces.to(self.device)  # same for all the gender
        else:
            self.smpl = SMPL_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                                model_root=str(model_root), hands=hands).to(self.device) 
            self.faces = self.smpl.th_faces.to(self.device) # XH: no need to input face, it is loaded from model file

        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = \
            torch_pose_obj_data(self.model_root, batch_size=batch_sz)


    def forward(self):
        if type(self.smpl) is list:
            verts_list, jtr_list, tposed_list, naked_list = [], [], [], []
            for i, smpl in enumerate(self.smpl):
                verts_i, jtr_i, tposed_i, naked_i = smpl(self.pose[i].unsqueeze(0),
                                                        th_betas=self.betas[i].unsqueeze(0),
                                                        th_trans=self.trans[i].unsqueeze(0),
                                                        th_offsets=self.offsets[i].unsqueeze(0))
                verts_list.append(verts_i)
                jtr_list.append(jtr_i)
                tposed_list.append(tposed_i)
                naked_list.append(naked_i)
            
            verts = torch.cat(verts_list, axis=0)
            jtr = torch.cat(jtr_list, axis=0)
            tposed = torch.cat(tposed_list, axis=0)
            naked = torch.cat(naked_list, axis=0)
            
        else:
            verts, jtr, tposed, naked = self.smpl(self.pose,
                                                th_betas=self.betas,
                                                th_trans=self.trans,
                                                th_offsets=self.offsets)
        return verts, jtr, tposed, naked

    def get_landmarks(self):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""
        if type(self.smpl) is list:
            verts_list = []
            for i, smpl in enumerate(self.smpl):
                verts_i, _, _, _ = smpl(self.pose[i].unsqueeze(0),
                                        th_betas=self.betas[i].unsqueeze(0),
                                        th_trans=self.trans[i].unsqueeze(0),
                                        th_offsets=self.offsets[i].unsqueeze(0))
                
                verts_list.append(verts_i)
            verts = torch.cat(verts_list, axis=0)
        else:   
            verts, _, _, _ = self.smpl(self.pose,
                                    th_betas=self.betas,
                                    th_trans=self.trans,
                                    th_offsets=self.offsets)

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)

        return J, face, hands

FEET_POSE_INDEX = [20,23,27,28,29,30,31,32]
BODY_POSE_INDEX = list(set(range(63))-set(FEET_POSE_INDEX))

class SMPLPyTorchWrapperBatchSplitParams(nn.Module):
    """
    Alternate implementation of SMPLPyTorchWrapperBatch that allows us to independently optimise:
     1. global_pose
     2. body pose (63 numbers)
     3. hand pose (6 numbers for SMPL or 90 numbers for SMPLH)
     4. top betas (primarily adjusts bone lengths)
     5. other betas
    """

    def __init__(self, model_root, batch_sz,
                 top_betas=None,
                 other_betas=None,
                 global_pose=None,
                 body_pose=None,
                 hand_pose=None,
                 trans=None,
                 offsets=None,
                 faces=None,
                 gender='male',
                 hands=False,
                 num_betas=300,
                 fixed_feet=False,
                 device='cuda:0'):
        super(SMPLPyTorchWrapperBatchSplitParams, self).__init__()
        self.model_root = model_root
        self.fixed_feet = fixed_feet

        if top_betas is None:
            self.top_betas = nn.Parameter(torch.zeros(batch_sz, TOP_BETA_NUM))
        else:
            assert top_betas.ndim == 2
            self.top_betas = nn.Parameter(top_betas)
        if other_betas is None:
            self.other_betas = nn.Parameter(torch.zeros(batch_sz, num_betas - TOP_BETA_NUM))
        else:
            assert other_betas.ndim == 2
            self.other_betas = nn.Parameter(other_betas)

        if global_pose is None:
            self.global_pose = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert global_pose.ndim == 2
            self.global_pose = nn.Parameter(global_pose)
        # if other_pose is None:
        #     self.other_pose = nn.Parameter(torch.zeros(batch_sz, 69))
        # else:
        #     assert other_pose.ndim == 2
        #     self.other_pose = nn.Parameter(other_pose)
        if body_pose is None:
            if fixed_feet:  
                self.body_pose = nn.Parameter(torch.zeros(batch_sz, len(BODY_POSE_INDEX)))  # BODY_POSE_NUM (63) - Feet (8)
            else:
                self.body_pose = nn.Parameter(torch.zeros(batch_sz, BODY_POSE_NUM)) 
        else:
            assert body_pose.ndim == 2
            if fixed_feet:
                self.body_pose = nn.Parameter(body_pose[:, BODY_POSE_INDEX])
            else:
                self.body_pose = nn.Parameter(body_pose)
        hand_pose_num = HAND_POSE_NUM if hands else SMPL_HAND_POSE_NUM
        if hand_pose is None:
            self.hand_pose = nn.Parameter(torch.zeros(batch_sz, hand_pose_num))
        else:
            assert hand_pose.ndim == 2
            assert hand_pose.shape[
                       1] == hand_pose_num, f'given hand pose dim {hand_pose.shape} does not match target model hand pose num of {hand_pose_num}'
            self.hand_pose = nn.Parameter(hand_pose)

        if trans is None:
            self.trans = nn.Parameter(torch.zeros(batch_sz, 3))
        else:
            assert trans.ndim == 2
            self.trans = nn.Parameter(trans)

        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(batch_sz, 6890, 3))
        else:
            assert offsets.ndim == 3
            self.offsets = nn.Parameter(offsets)

        self.betas = torch.cat([self.top_betas.clone(), self.other_betas.clone()], axis=1)

        if self.fixed_feet:
            self.pose = torch.cat([self.global_pose.clone(), 
                                self.body_pose[:,:20].clone(), 
                                torch.zeros(batch_sz, 1).to(device),
                                self.body_pose[:,20:22].clone(), 
                                torch.zeros(batch_sz, 1).to(device),
                                self.body_pose[:,22:25].clone(), 
                                torch.zeros(batch_sz, 6).to(device),
                                self.body_pose[:,25:].clone(), 
                                self.hand_pose.clone()], axis=1)
        else:
            self.pose = torch.cat([self.global_pose.clone(), self.body_pose.clone(), self.hand_pose.clone()], axis=1)
        # self.pose = torch.cat([self.global_pose.clone(), 
        #                        self.body_pose[:,:20].clone(), 
        #                        torch.zeros(batch_sz, 1).to(device),
        #                        self.body_pose[:,20:22].clone(), 
        #                        torch.zeros(batch_sz, 1).to(device),
        #                        self.body_pose[:,22:26].clone(), 
        #                        torch.zeros(batch_sz, 2).to(device),
        #                        self.body_pose[:,[26]].clone(), 
        #                        torch.zeros(batch_sz, 2).to(device),
        #                        self.body_pose[:,27:].clone(), 
        #                        self.hand_pose.clone()], axis=1)
        self.faces = faces
        self.gender = gender

        # pytorch smpl
        if type(gender) is list:
            self.smpl = []
            for g in gender:
                self.smpl.append(SMPL_Layer(center_idx=0, gender=g, num_betas=num_betas,
                                model_root=str(model_root), hands=hands).to(device))
        else:
            self.smpl = SMPL_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                                model_root=str(model_root), hands=hands).to(device)

        # Landmarks
        self.body25_reg_torch, self.face_reg_torch, self.hand_reg_torch = \
            torch_pose_obj_data(self.model_root, batch_size=batch_sz)

    def forward(self):

        self.betas = torch.cat([self.top_betas.clone(), self.other_betas.clone()], axis=1)
        if self.fixed_feet:
            batch_sz = self.betas.shape[0]
            device = self.betas.device
            self.pose = torch.cat([self.global_pose.clone(), 
                                self.body_pose[:,:20].clone(), 
                                torch.zeros(batch_sz, 1).to(device),
                                self.body_pose[:,20:22].clone(), 
                                torch.zeros(batch_sz, 1).to(device),
                                self.body_pose[:,22:25].clone(), 
                                torch.zeros(batch_sz, 6).to(device),
                                self.body_pose[:,25:].clone(), 
                                self.hand_pose.clone()], axis=1)
        else:
            self.pose = torch.cat([self.global_pose.clone(), self.body_pose.clone(), self.hand_pose.clone()], axis=1)
        # self.pose = torch.cat([self.global_pose.clone(), 
        #                        self.body_pose[:,:20].clone(), 
        #                        torch.zeros(batch_sz, 1).to(device),
        #                        self.body_pose[:,20:22].clone(), 
        #                        torch.zeros(batch_sz, 1).to(device),
        #                        self.body_pose[:,22:26].clone(), 
        #                        torch.zeros(batch_sz, 2).to(device),
        #                        self.body_pose[:,[26]].clone(), 
        #                        torch.zeros(batch_sz, 2).to(device),
        #                        self.body_pose[:,27:].clone(), 
        #                        self.hand_pose.clone()], axis=1)

        if type(self.smpl) is list:
            verts_list, jtr_list, tposed_list, naked_list = [], [], [], []
            for i, smpl in enumerate(self.smpl):
                verts_i, jtr_i, tposed_i, naked_i = smpl(self.pose[i].unsqueeze(0),
                                                        th_betas=self.betas[i].unsqueeze(0),
                                                        th_trans=self.trans[i].unsqueeze(0),
                                                        th_offsets=self.offsets[i].unsqueeze(0))
                verts_list.append(verts_i)
                jtr_list.append(jtr_i)
                tposed_list.append(tposed_i)
                naked_list.append(naked_i)
            
            verts = torch.cat(verts_list, axis=0)
            jtr = torch.cat(jtr_list, axis=0)
            tposed = torch.cat(tposed_list, axis=0)
            naked = torch.cat(naked_list, axis=0)
            
        else:
            verts, jtr, tposed, naked = self.smpl(self.pose,
                                                th_betas=self.betas,
                                                th_trans=self.trans,
                                                th_offsets=self.offsets)
        return verts, jtr, tposed, naked

    def get_landmarks(self):
        """Computes body25 joints for SMPL along with hand and facial landmarks"""
        
        self.betas = torch.cat([self.top_betas.clone(), self.other_betas.clone()], axis=1)
        if self.fixed_feet:
            batch_sz = self.betas.shape[0]
            device = self.betas.device
            self.pose = torch.cat([self.global_pose.clone(), 
                                self.body_pose[:,:20].clone(), 
                                torch.zeros(batch_sz, 1).to(device),
                                self.body_pose[:,20:22].clone(), 
                                torch.zeros(batch_sz, 1).to(device),
                                self.body_pose[:,22:25].clone(), 
                                torch.zeros(batch_sz, 6).to(device),
                                self.body_pose[:,25:].clone(), 
                                self.hand_pose.clone()], axis=1)
        else:
            self.pose = torch.cat([self.global_pose.clone(), self.body_pose.clone(), self.hand_pose.clone()], axis=1)
        # self.pose = torch.cat([self.global_pose.clone(), 
        #                        self.body_pose[:,:20].clone(), 
        #                        torch.zeros(batch_sz, 1).to(device),
        #                        self.body_pose[:,20:22].clone(), 
        #                        torch.zeros(batch_sz, 1).to(device),
        #                        self.body_pose[:,22:26].clone(), 
        #                        torch.zeros(batch_sz, 2).to(device),
        #                        self.body_pose[:,[26]].clone(), 
        #                        torch.zeros(batch_sz, 2).to(device),
        #                        self.body_pose[:,27:].clone(), 
        #                        self.hand_pose.clone()], axis=1)

        if type(self.smpl) is list:
            verts_list = []
            for i, smpl in enumerate(self.smpl):
                verts_i, _, _, _ = smpl(self.pose[i].unsqueeze(0),
                                        th_betas=self.betas[i].unsqueeze(0),
                                        th_trans=self.trans[i].unsqueeze(0),
                                        th_offsets=self.offsets[i].unsqueeze(0))
                
                verts_list.append(verts_i)
            verts = torch.cat(verts_list, axis=0)
            
        else:
            verts, _, _, _ = self.smpl(self.pose,
                                        th_betas=self.betas,
                                        th_trans=self.trans,
                                        th_offsets=self.offsets)

        J = batch_sparse_dense_matmul(self.body25_reg_torch, verts)
        face = batch_sparse_dense_matmul(self.face_reg_torch, verts)
        hands = batch_sparse_dense_matmul(self.hand_reg_torch, verts)

        return J, face, hands

    @staticmethod
    def from_smpl(smpl: SMPLPyTorchWrapperBatch, fixed_feet=False):
        """
        construct a split smpl from a smpl module
        Args:
            smpl:

        Returns:

        """
        batch_sz = smpl.pose.shape[0]
        split_smpl = SMPLPyTorchWrapperBatchSplitParams(smpl.model_root,
                                                         batch_sz,
                                                         trans=smpl.trans.data,
                                                         top_betas=smpl.betas.data[:, :TOP_BETA_NUM],
                                                         other_betas=smpl.betas.data[:, TOP_BETA_NUM:],
                                                         global_pose=smpl.pose.data[:, :GLOBAL_POSE_NUM],
                                                         body_pose=smpl.pose.data[:, GLOBAL_POSE_NUM:GLOBAL_POSE_NUM + BODY_POSE_NUM],
                                                         hand_pose=smpl.pose.data[:, GLOBAL_POSE_NUM + BODY_POSE_NUM:],
                                                         faces=smpl.faces, gender=smpl.gender,
                                                         hands=smpl.hands,
                                                         fixed_feet=fixed_feet,
                                                         device=smpl.device)
        return split_smpl


class SMPLPyTorchWrapper(nn.Module):
    "XH: this one is not used, why keeping it?"
    def __init__(self, model_root, betas=None, pose=None, trans=None, offsets=None, gender='male', num_betas=300):
        super(SMPLPyTorchWrapper, self).__init__()
        if betas is None:
            self.betas = nn.Parameter(torch.zeros(300,))
        else:
            self.betas = nn.Parameter(betas)
        if pose is None:
            self.pose = nn.Parameter(torch.zeros(72,))
        else:
            self.pose = nn.Parameter(pose)
        if trans is None:
            self.trans = nn.Parameter(torch.zeros(3,))
        else:
            self.trans = nn.Parameter(trans)
        if offsets is None:
            self.offsets = nn.Parameter(torch.zeros(6890, 3))
        else:
            self.offsets = nn.Parameter(offsets)

        ## pytorch smpl
        self.smpl = SMPL_Layer(center_idx=0, gender=gender, num_betas=num_betas,
                               model_root=str(model_root))

    def forward(self):
        verts, Jtr, tposed, naked = self.smpl(self.pose.unsqueeze(axis=0),
                                              th_betas=self.betas.unsqueeze(axis=0),
                                              th_trans=self.trans.unsqueeze(axis=0),
                                              th_offsets=self.offsets.unsqueeze(axis=0))
        return verts[0]
