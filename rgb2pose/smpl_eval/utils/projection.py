import numpy as np
import torch

def project_to_2d(data_3d, proj_matrix):
    """
    Project 3d coordinates to 2d
    data_3d: (B, N, 3)
    proj_matrix: (3, 4)

    return
    data_2d: (B, N, 2)
    """
    
    ext_arr = np.ones((data_3d.shape[0], data_3d.shape[1], 1))
    data_homo = np.concatenate((data_3d, ext_arr), axis=2)
    temp = data_homo @ proj_matrix.T 
    data_2d = temp[:,:,:2]/temp[:,:,[2]]

    return data_2d

def word2cam(data_world, proj_matrix):
    """
    Project 3d world coordinates to 3d camera coordinates
    data_world: (B, N, 3)
    proj_matrix: (3, 4)

    return
    data_cam: (B, N, 3)
    """
    
    ext_arr = np.ones((data_world.shape[0], data_world.shape[1], 1))
    data_homo = np.concatenate((data_world, ext_arr), axis=2)
    data_cam = data_homo @ proj_matrix.T 

    return data_cam