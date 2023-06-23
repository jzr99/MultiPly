import torch
from torch import nn
import numpy as np
from torch.nn import functional as F

class Loss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.eikonal_weight = opt.eikonal_weight
        self.bce_weight = opt.bce_weight
        self.opacity_sparse_weight = opt.opacity_sparse_weight
        self.in_shape_weight = opt.in_shape_weight
        self.sam_mask_weight = opt.sam_mask_weight
        self.eps = 1e-6
        self.milestone = 200
        self.sam_milestone = 1000
        self.smpl_surface_milestone = opt.smpl_surface_milestone
        self.depth_loss_milestone = 1000
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.l2_loss = nn.MSELoss(reduction='mean')
        self.sigmoid = nn.Sigmoid()
    
    # L1 reconstruction loss for RGB values
    def get_rgb_loss(self, rgb_values, rgb_gt):
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)
        return rgb_loss
    
    # Eikonal loss introduced in IGR
    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
        return eikonal_loss

    # BCE loss for clear boundary
    def get_bce_los(self, acc_map):
        binary_loss = -1 * (acc_map * (acc_map + self.eps).log() + (1-acc_map) * (1 - acc_map + self.eps).log()).mean() * 2
        return binary_loss

    # Global opacity sparseness regularization 
    def get_opacity_sparse(self, acc_map, index_off_surface):
        opacity_sparse_loss = self.l1_loss(acc_map[index_off_surface], torch.zeros_like(acc_map[index_off_surface]))
        return opacity_sparse_loss

    # Optional: This loss helps to stablize the training in the very beginning
    def get_in_shape_loss(self, acc_map, index_in_surface):
        in_shape_loss = self.l1_loss(acc_map[index_in_surface], torch.ones_like(acc_map[index_in_surface]))
        return in_shape_loss

    def get_sam_mask_loss(self, sam_mask, acc_person):
        sam_mask = self.sigmoid(sam_mask)
        valid_mask = sam_mask.sum(dim=1) <= (1 + 100 * self.eps)
        loss = self.l1_loss(acc_person[valid_mask], sam_mask[valid_mask])
        # import pdb;pdb.set_trace()
        return loss

    def get_depth_order_loss(self, t_list, fg_rgb_values_each_person_list_nan_filter, mean_hitted_vertex_list, rgb_gt_nan_filter_hitted, cam_loc):
        front_person_index = np.argmin(t_list, axis=0)
        # todo check fg_rgb_values_each_person_list_nan_filter (number_person, number_pixel, 3)
        rgb_loss = torch.norm(fg_rgb_values_each_person_list_nan_filter - rgb_gt_nan_filter_hitted.reshape(1, -1, 3), dim=-1)
        correct_rgb_person_index = torch.argmin(rgb_loss, dim=0)
        # mean_hitted_vertex_list shape (2, 7, 3)
        d1, d2, d3= mean_hitted_vertex_list.shape
        # import pdb;pdb.set_trace()
        front_vertex_position = mean_hitted_vertex_list[front_person_index, torch.arange(d2), :]
        correct_vertex_position = mean_hitted_vertex_list[correct_rgb_person_index, torch.arange(d2), :]
        dist_correct = torch.norm(correct_vertex_position - cam_loc, dim=-1)
        dist_front = torch.norm(front_vertex_position - cam_loc, dim=-1)
        loss = torch.log(1+torch.exp(dist_correct - dist_front)).sum()
        return loss
        # front_vertex_position = front_vertex_position[:, torch.arange(d2)]

    def get_depth_order_loss_samGT(self, t_list, mean_hitted_vertex_list, sam_mask, cam_loc):
        front_person_index = np.argmin(t_list, axis=0)
        # check sam_mask shape (number_pixel, number_person)
        correct_rgb_person_index = np.argmax(sam_mask.cpu().numpy(), axis=1)
        d1, d2, d3 = mean_hitted_vertex_list.shape
        # import pdb;pdb.set_trace()
        front_vertex_position = mean_hitted_vertex_list[front_person_index, torch.arange(d2), :]
        correct_vertex_position = mean_hitted_vertex_list[correct_rgb_person_index, torch.arange(d2), :]
        dist_correct = torch.norm(correct_vertex_position - cam_loc, dim=-1)
        dist_front = torch.norm(front_vertex_position - cam_loc, dim=-1)
        loss = torch.log(1 + torch.exp(dist_correct - dist_front)).sum()
        return loss

    def forward(self, model_outputs, ground_truth):
        if isinstance(model_outputs['fg_rgb_values_each_person_list'], list):
            depth_order_loss = torch.zeros((1),device=model_outputs['acc_map'].device)
        else:
            fg_rgb_values_each_person_list = model_outputs['fg_rgb_values_each_person_list']
            fg_rgb_values_each_person_nan_filter = ~torch.any(torch.any(fg_rgb_values_each_person_list.isnan(), dim=-1), dim=0)
            fg_rgb_values_each_person_list_nan_filter = fg_rgb_values_each_person_list[:, fg_rgb_values_each_person_nan_filter, :]
            gt_rgb_each_person = ground_truth['rgb'][0].cuda()
            gt_rgb_each_person = gt_rgb_each_person[model_outputs["hitted_mask_idx"]]
            rgb_gt_nan_filter_hitted = gt_rgb_each_person[fg_rgb_values_each_person_nan_filter]
            cam_loc = model_outputs['cam_loc'][model_outputs["hitted_mask_idx"]]
            cam_loc = cam_loc[fg_rgb_values_each_person_nan_filter]
            # import pdb;pdb.set_trace()

            # reorder GT sam mask
            if 'sam_mask' in model_outputs.keys() and True:
                sam_mask = model_outputs['sam_mask']
                sam_mask_reorder = sam_mask[model_outputs["hitted_mask_idx"]]
                sam_mask_reorder_filter = sam_mask_reorder[fg_rgb_values_each_person_nan_filter]
                depth_order_loss = self.get_depth_order_loss_samGT(model_outputs['t_list'],
                                                             model_outputs['mean_hitted_vertex_list'],
                                                             sam_mask_reorder_filter, cam_loc)
            else:
                print('using rgb evidence depth order loss')
                depth_order_loss = self.get_depth_order_loss(model_outputs['t_list'], fg_rgb_values_each_person_list_nan_filter, model_outputs['mean_hitted_vertex_list'], rgb_gt_nan_filter_hitted, cam_loc)

        nan_filter = ~torch.any(model_outputs['rgb_values'].isnan(), dim=1)
        rgb_gt = ground_truth['rgb'][0].cuda()
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'][nan_filter], rgb_gt[nan_filter])
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        bce_loss = self.get_bce_los(model_outputs['acc_map'])
        opacity_sparse_loss = self.get_opacity_sparse(model_outputs['acc_map'], model_outputs['index_off_surface'])
        in_shape_loss = self.get_in_shape_loss(model_outputs['acc_map'], model_outputs['index_in_surface'])
        curr_epoch_for_loss = min(self.milestone, model_outputs['epoch']) # will not increase after the milestone
        interpenetration_loss = model_outputs['interpenetration_loss']
        temporal_loss = model_outputs['temporal_loss']
        smpl_surface_loss = model_outputs['smpl_surface_loss']
        if 'sam_mask' in model_outputs.keys() and model_outputs['epoch'] > 200:
            sam_mask_loss = self.get_sam_mask_loss(model_outputs['sam_mask'], model_outputs['acc_person_list'])
        else:
            sam_mask_loss = torch.zeros((1),device=in_shape_loss.device)
        # if model_outputs['epoch'] > 300:
        if model_outputs['epoch'] > 200:
            depth_order_loss = 1.0 * depth_order_loss * (1 - min(self.depth_loss_milestone, model_outputs['epoch']) / self.depth_loss_milestone)
        else:
            depth_order_loss = torch.zeros((1), device=model_outputs['acc_map'].device)
        loss = rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.bce_weight * bce_loss + \
               self.opacity_sparse_weight * (1 + curr_epoch_for_loss ** 2 / 40) * opacity_sparse_loss + \
               self.in_shape_weight * (1 - curr_epoch_for_loss / self.milestone) * in_shape_loss + \
               interpenetration_loss + temporal_loss + \
               self.sam_mask_weight * sam_mask_loss + \
               smpl_surface_loss * (1 - min(self.smpl_surface_milestone, model_outputs['epoch']) / self.smpl_surface_milestone) + \
               depth_order_loss
               # self.sam_mask_weight * (1 - min(self.sam_milestone, model_outputs['epoch']) / self.sam_milestone) * sam_mask_loss
        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'depth_order_loss': depth_order_loss,
            'eikonal_loss': eikonal_loss,
            'bce_loss': bce_loss,
            'opacity_sparse_loss': opacity_sparse_loss,
            'in_shape_loss': in_shape_loss,
            'interpenetration_loss': interpenetration_loss,
            'temporal_loss': temporal_loss,
            'sam_mask_loss': sam_mask_loss,
            'smpl_surface_loss': smpl_surface_loss,
        }