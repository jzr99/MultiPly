import os
import cv2
import numpy as np
import glob
import trimesh
import pickle as pkl
import tqdm
seq = 'outdoors_fencing_01'
if seq == 'outdoors_fencing_01':
    start_idx = 546

DIR = '/home/chen/RGB-PINA/code/outputs/ThreeDPW'
save_dir = f'{DIR}/{seq}_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
mesh_paths = sorted(glob.glob(f'{DIR}/{seq}_wo_disp_freeze_20_every_20_opt_pose/test_mesh/*_deformed.ply'))
cam_path = f'/home/chen/RGB-PINA/data/{seq}/cameras_normalize.npz'
cam = dict(np.load(cam_path))

seq_dir = f'/home/chen/disk2/3DPW/sequenceFiles/test/{seq}.pkl'
seq_file = pkl.load(open(seq_dir, 'rb'), encoding='latin1')


for idx, mesh_path in enumerate(mesh_paths):
    scaled_mesh = trimesh.load(mesh_path, process=False)
    scaling_factor = cam[f'scale_mat_{idx}'][0, 0]
    scaled_mesh.apply_scale(scaling_factor)
    cam_P = cam[f'world_mat_{idx}']
    out = cv2.decomposeProjectionMatrix(cam_P[:3, :])
    cam_R = out[1]
    cam_center = out[2]
    cam_center = (cam_center[:3] / cam_center[3])[:, 0]

    cam_T = -cam_R @ cam_center
    cam_extrinsic = np.eye(4)
    cam_extrinsic[:3, :3] = cam_R
    cam_extrinsic[:3, 3] = cam_T

    original_cam_extrinsic = seq_file['cam_poses'][idx + start_idx]

    scaled_mesh.apply_transform(np.linalg.inv(original_cam_extrinsic) @ cam_extrinsic)
    _ = scaled_mesh.export(os.path.join(save_dir, os.path.basename(mesh_path)))
    
