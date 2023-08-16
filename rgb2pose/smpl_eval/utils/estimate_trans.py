import cv2
import numpy as np
import torch

def estimate_translation_cv2(joints_3d, joints_2d, proj_mat=None, cam_dist=None):
    camK = proj_mat
    ret, rvec, tvec,inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                              flags=cv2.SOLVEPNP_EPNP,reprojectionError=20,iterationsCount=100)
    if inliers is None:
        return None
    else:
        tra_pred = tvec[:,0]              
        return tra_pred

def estimate_translation(joints_3d, joints_2d, org_trans, proj_mats=None, cam_dists=None):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """
    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float)
    if cam_dists is None:
        cam_dists = [None for _ in range(len(joints_2d))]
    # Find the translation for each example in the batch
    for i in range(joints_3d.shape[0]):
        trans_i = estimate_translation_cv2(joints_3d[i], joints_2d[i], 
                proj_mat=proj_mats, cam_dist=cam_dists[i])
        trans[i] = trans_i if trans_i is not None else org_trans[i]

    return torch.from_numpy(trans).float()

def depth_relation(depth_dist, dr_thresh=0.1):
    if np.abs(depth_dist) < dr_thresh:
        return "equal"
    elif depth_dist >= dr_thresh:
        return "back"
    else:
        return "front"