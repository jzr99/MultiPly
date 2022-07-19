from torch import unsqueeze
from utils import GMoF, MaxMixturePrior
import torch
num_joints = 25
joints_to_ign = [1,9,12]
joint_weights = torch.ones(num_joints)   
joint_weights[joints_to_ign] = 0
joint_weights = joint_weights.reshape((-1,1)).cuda()

robustifier = GMoF(rho=100)
body_pose_prior = MaxMixturePrior().cuda()

def get_loss_weights():
    loss_weight = {'J2D_Loss': lambda cst, it: 1e-2 * cst, # / (1 + 1 * it),
                   'Prior_Loss': lambda cst, it: 2e-3 * cst, # / (1 + it), # 1e-4 for BodyFusion else 2e-5 for others
                   'Prior_Shape': lambda cst, it: 1. * cst, # / (1 + it),
                   'Temporal_Loss': lambda cst, it: 2e0 * cst,
                  }
    return loss_weight

def joints_2d_loss(gt_joints_2d=None, joints_2d=None, joint_confidence=None):

    joint_diff = robustifier(gt_joints_2d - joints_2d)
    joints_2dloss = torch.mean((joint_confidence*joint_weights[:, 0]).unsqueeze(-1) ** 2 * joint_diff)
    return joints_2dloss

def pose_prior_loss(body_pose=None, betas=None):
    pprior_loss = torch.sum(body_pose_prior(body_pose, betas))
    return pprior_loss

def pose_temporal_loss(last_pose, param_pose):
    temporal_loss = torch.mean(torch.square(last_pose - param_pose))
    return temporal_loss