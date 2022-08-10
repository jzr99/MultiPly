import torch
from torch import nn
from torch.nn import functional as F


class IDRLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.eikonal_weight = opt.eikonal_weight
        self.mask_weight = opt.mask_weight
        self.bone_weight = opt.bone_weight
        self.reg_weight = opt.reg_weight
        self.normal_weight = opt.normal_weight
        self.alpha = opt.alpha
        self.l1_loss = nn.L1Loss(reduction='sum')
        self.l2_loss = nn.MSELoss(reduction='sum')

    def get_rgb_loss(self, rgb_values, rgb_gt, network_object_mask,
                     object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(
            object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
        return eikonal_loss

    def get_lbs_loss(self, lbs_weight, gt_lbs_weight, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        lbs_weight = lbs_weight[network_object_mask & object_mask]
        gt_lbs_weight = gt_lbs_weight[network_object_mask & object_mask]
        lbs_loss =self.l2_loss(lbs_weight, gt_lbs_weight)/ float(object_mask.shape[0])
        return lbs_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_normal_loss(self, normal_values, surface_normal_gt, network_object_mask, object_mask, normal_weight):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()
        
        normal_values = normal_values[network_object_mask & object_mask]
        normal_weight = normal_weight[network_object_mask & object_mask]
        normal_loss = torch.sum(normal_weight[:, None] * ((normal_values-surface_normal_gt) ** 2)) / float(object_mask.shape[0])
        return normal_loss

    def forward(self, model_outputs, ground_truth):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        # nan_filter = ~torch.any(model_outputs['rgb_values'].isnan(), dim=1)
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt,
                                     network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'],
                                       network_object_mask, object_mask)
        bone_loss = F.mse_loss(model_outputs['w_pd'], model_outputs['w_gt'])
        lbs_loss = self.get_lbs_loss(model_outputs['lbs_weight_pd'], model_outputs['lbs_weight_gt'], network_object_mask, object_mask)
        normal_loss = self.get_normal_loss(model_outputs['normal_values'], model_outputs['surface_normal_gt'], network_object_mask, object_mask, model_outputs['normal_weight'])
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])

        # loss = rgb_loss + \
        #        self.eikonal_weight * eikonal_loss + \
        #        self.mask_weight * mask_loss

        #    self.bone_weight * bone_loss
        
        loss = rgb_loss + self.mask_weight * mask_loss + self.reg_weight * lbs_loss + self.bone_weight * bone_loss + self.normal_weight * normal_loss + self.eikonal_weight * eikonal_loss 

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'bone_loss': bone_loss,
            'lbs_loss': lbs_loss,
            'normal_loss': normal_loss,
        }
    
class VolSDFLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.eikonal_weight = opt.eikonal_weight
        self.bone_weight = opt.bone_weight
        self.normal_loss_weight = opt.norm_weight
        self.density_reg_weight = 5e-3
        self.bg_shadow_weight = 5e-2
        self.off_surface_weight = 3e-3
        self.in_surface_weight = 1e-2
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
        
    def get_normal_loss(self, normal_values, surface_normal_gt, normal_weight, acc_map):
        # TODO Check
        normal_loss = torch.mean(normal_weight[:, None] * acc_map[:, None] * torch.norm((normal_values-surface_normal_gt) ** 2, dim=1)) # torch.mean(torch.norm((normal_values-surface_normal_gt) ** 2, dim=1)) # 
        return normal_loss
    
    def get_bone_loss(self, w_pd, w_gt):
        bone_loss = self.l2_loss(w_pd, w_gt)
        return bone_loss
    def get_foot_reg_loss(self, foot_sample_sdf_pd, foot_sample_sdf_gt):
        foot_reg_loss = self.l2_loss(foot_sample_sdf_pd, foot_sample_sdf_gt)
        return foot_reg_loss

    def get_density_reg_loss(self, acc_map, index_outside):
        zero_density_loss = self.l1_loss(acc_map, torch.zeros_like(acc_map)) 
        outside_bbox_density_loss = self.l1_loss(acc_map[index_outside[0]], torch.zeros_like(acc_map[index_outside[0]]))
        # ground_density_loss = self.l1_loss(acc_map[index_ground[:, 0]], torch.zeros_like(acc_map[index_ground[:, 0]]))
        binary_loss = -1 * (acc_map * (acc_map + 1e-6).log() + (1-acc_map) * (1 - acc_map + 1e-6).log()).mean() 
        density_reg_loss = 0.5 * zero_density_loss + 5 * outside_bbox_density_loss + 2 * binary_loss # + 0.1 * ground_density_loss
        return density_reg_loss

    def get_off_surface_loss(self, acc_map, index_off_surface):
        off_surface_loss = self.l1_loss(acc_map[index_off_surface], torch.zeros_like(acc_map[index_off_surface]))  # torch.sum(acc_map[index_off_surface])
        return off_surface_loss

    def get_in_surface_loss(self, acc_map, index_in_surface):
        in_surface_loss = self.l1_loss(acc_map[index_in_surface], torch.ones_like(acc_map[index_in_surface]))
        return in_surface_loss

    def get_bg_shadow_loss(self, bg_rgb_values):
        bg_shadow_loss = torch.mean(bg_rgb_values)
        return bg_shadow_loss

    def forward(self, model_outputs, ground_truth):
        nan_filter = ~torch.any(model_outputs['rgb_values'].isnan(), dim=1)
        rgb_gt = ground_truth['rgb'][0].cuda()
        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'][nan_filter], rgb_gt[nan_filter])
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        density_reg_loss = self.get_density_reg_loss(model_outputs['acc_map'], model_outputs['index_outside'])
        off_surface_loss = self.get_off_surface_loss(model_outputs['acc_map'], model_outputs['index_off_surface'])
        in_surface_loss = self.get_in_surface_loss(model_outputs['acc_map'], model_outputs['index_in_surface'])
        epoch_for_off_surface = min(200, model_outputs['epoch'])
        if model_outputs['epoch'] >= 10000:
            normal_loss = self.get_normal_loss(model_outputs['normal_values'], model_outputs['surface_normal_gt'], model_outputs['normal_weight'], model_outputs['acc_map'])
            if model_outputs['use_smpl_deformer']:
                loss = rgb_loss + self.eikonal_weight * eikonal_loss + self.density_reg_weight * density_reg_loss + self.off_surface_weight * (1 + epoch_for_off_surface ** 2 / 40) * off_surface_loss + self.normal_loss_weight * normal_loss
                return {
                    'loss': loss,
                    'rgb_loss': rgb_loss,
                    'eikonal_loss': eikonal_loss,
                    'density_reg_loss': density_reg_loss,
                    'normal_loss': normal_loss,
                    'off_surface_loss': off_surface_loss,
                }
        # bg_shadow_loss = self.get_bg_shadow_loss(model_outputs['bg_rgb_values'])
        # foot_reg_loss = self.get_foot_reg_loss(model_outputs['foot_sample_sdf_pd'], model_outputs['foot_sample_sdf_gt'])
        
        else:
            if model_outputs['use_smpl_deformer']:
                loss = rgb_loss + self.eikonal_weight * eikonal_loss + self.density_reg_weight * density_reg_loss + self.off_surface_weight * (1 + epoch_for_off_surface ** 2 / 40) * off_surface_loss + \
                       self.in_surface_weight * in_surface_loss
                return {
                    'loss': loss,
                    'rgb_loss': rgb_loss,
                    'eikonal_loss': eikonal_loss,
                    'density_reg_loss': density_reg_loss,
                    # 'normal_loss': normal_loss,
                    'off_surface_loss': off_surface_loss,
                    'in_surface_loss': in_surface_loss,
                }
            else:
                bone_loss = self.get_bone_loss(model_outputs['w_pd'], model_outputs['w_gt'])
                loss = rgb_loss + self.eikonal_weight * eikonal_loss + self.bone_weight * bone_loss + self.normal_weight * normal_loss 
                return {
                    'loss': loss,
                    'rgb_loss': rgb_loss,
                    'eikonal_loss': eikonal_loss,
                    'bone_loss': bone_loss,
                    'normal_loss': normal_loss,
                }

class ThreeDLoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.eikonal_weight = 0.1 # opt.eikonal_weight
        self.global_weight = 0.01 # opt.global_weight
        self.bone_weight = 1.
        self.l1_loss = nn.L1Loss(reduction='mean')
    
    def get_sdf_loss(self, sdf_output, sdf_gt):
        sdf_loss = self.l1_loss(sdf_output, sdf_gt)
        return sdf_loss
    
    def get_eikonal_loss(self, grad_theta):
        eikonal_loss = ((grad_theta.norm(2, dim=-1) - 1)**2).mean()
        return eikonal_loss
    
    def forward(self, model_outputs):
        on_surface_gt = model_outputs['on_surface_gt'].cuda()
        off_surface_gt = model_outputs['off_surface_gt'].cuda()

        on_surface_loss = self.get_sdf_loss(model_outputs['on_surface_sdf'], on_surface_gt)
        off_surface_loss = self.get_sdf_loss(model_outputs['off_surface_sdf'], off_surface_gt)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        bone_loss = F.mse_loss(model_outputs['w_pd'], model_outputs['w_gt'])
        loss = on_surface_loss + \
               self.global_weight * off_surface_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.bone_weight * bone_loss

        return {
            'loss': loss,
            'on_surface_loss': on_surface_loss,
            'off_surface_loss': off_surface_loss,
            'eikonal_loss': eikonal_loss,
            'bone_loss': bone_loss
        }