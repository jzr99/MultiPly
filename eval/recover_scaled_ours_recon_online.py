import os
import cv2
import numpy as np
import glob
import trimesh
import pickle as pkl
from tqdm import tqdm
from utils import rectify_pose, compute_similarity_transform, transform_mesh
import sys
sys.path.append('/home/chen/RGB-PINA/rgb2pose')
from smplx import SMPL

seq = 'Invisible'
gender = 'male'
if seq == 'Invisible':
    start_idx = 0 # 546

DIR = '/home/chen/RGB-PINA/code/outputs/ThreeDPW'
save_dir = f'{DIR}/{seq}_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
mesh_paths = sorted(glob.glob(f'{DIR}/{seq}_wo_disp_freeze_20_every_20_opt_pose/test_mesh/*_deformed.ply'))
gt_smpl_mesh_paths = sorted(glob.glob(f'/home/chen/disk2/3DPW_GT/{seq}/smpl_mesh/*.obj'))
cam_path = f'/home/chen/RGB-PINA/data/{seq}/cameras_normalize.npz'
cam = dict(np.load(cam_path))

estimated = True

import torch


smpl_model = SMPL('/home/chen/Models/smpl', gender=gender).cuda()
for idx, mesh_path in tqdm(enumerate(mesh_paths)):
    scaled_mesh = trimesh.load(mesh_path, process=False)
    scaling_factor = cam[f'scale_mat_{idx}'][0, 0]
    scaled_mesh.apply_scale(scaling_factor)


    _ = scaled_mesh.export(os.path.join(save_dir, os.path.basename(mesh_path)))
    
