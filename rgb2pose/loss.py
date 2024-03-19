from torch import unsqueeze
from utils import GMoF, MaxMixturePrior
import torch
robustifier = GMoF(rho=100)
body_pose_prior = MaxMixturePrior().cuda()

def get_loss_weights():
    loss_weight = {'J2D_Loss': lambda cst, it: 1e-2 * cst, # / (1 + 1 * it),
                   'Prior_Loss': lambda cst, it: 2e-3 * cst, # / (1 + it), # 1e-4 for BodyFusion else 2e-5 for others
                   'Prior_Shape': lambda cst, it: 1. * cst, # / (1 + it),
                   'Temporal_Loss': lambda cst, it: 6e0 * cst,
                   'FOOT_Prior_Loss': lambda cst, it: 1e-1 * cst,
                  }
    return loss_weight

def joints_2d_loss(gt_joints_2d=None, joints_2d=None, joint_confidence=None, is_vitpose=False):

    if is_vitpose:
        num_joints = 17
        # ign L/R hip
        joints_to_ign = [11,12]
        # joints_to_ign = []
        joint_weights = torch.ones(num_joints)   
        joint_weights[joints_to_ign] = 0
        joint_weights = joint_weights.reshape((-1,1)).cuda()
    else:
        num_joints = 25
        joints_to_ign = [1,9,12]
        joint_weights = torch.ones(num_joints)   
        joint_weights[joints_to_ign] = 0
        joint_weights = joint_weights.reshape((-1,1)).cuda()


    joint_diff = robustifier(gt_joints_2d - joints_2d)
    joints_2dloss = torch.mean((joint_confidence*joint_weights[:, 0]).unsqueeze(-1) ** 2 * joint_diff)
    return joints_2dloss

def pose_prior_loss(body_pose=None, betas=None):
    pprior_loss = torch.sum(body_pose_prior(body_pose, betas))
    return pprior_loss

def pose_temporal_loss(last_pose, param_pose):
    temporal_loss = torch.mean(torch.square(last_pose - param_pose))
    return temporal_loss

def foot_prior_loss(foot_pose=None):
    foot_prior_loss = torch.max((foot_pose)**2)
    return foot_prior_loss