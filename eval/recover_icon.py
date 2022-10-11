import os
import cv2
import numpy as np
import glob
import trimesh
import pickle as pkl
import tqdm
seq = 'outdoors_fencing_01'
if seq == 'outdoors_fencing_01':
    start_idx = 0 # 546

DIR = '/home/chen/ICON_new'
save_dir = f'{DIR}/{seq}_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
mesh_paths = sorted(glob.glob(f'{DIR}/{seq}/icon-filter/obj/*_recon.obj'))
smpl_paths = sorted(glob.glob(f'{DIR}/{seq}/icon-filter/obj/*_smpl.npy'))
cam_path = f'/home/chen/RGB-PINA/data/{seq}/cameras_normalize.npz'
cam = dict(np.load(cam_path))

seq_dir = '/home/chen/disk2/3DPW/sequenceFiles/test/outdoors_fencing_01.pkl' # f'/home/chen/disk2/3DPW/sequenceFiles/test/{seq}.pkl'
seq_file = pkl.load(open(seq_dir, 'rb'), encoding='latin1')

estimated = True
resize_factor = 2

original_K = np.eye(4) 
original_K[:3, :3] = seq_file['cam_intrinsics']
original_K[0, 0] = original_K[0, 0] / resize_factor
original_K[1, 1] = original_K[1, 1] / resize_factor
original_K[0, 2] = original_K[0, 2] / resize_factor
original_K[1, 2] = original_K[1, 2] / resize_factor
for idx, mesh_path in enumerate(mesh_paths):
    scaled_mesh = trimesh.load(mesh_path, process=False)

    cam_P = cam[f'world_mat_{idx}']
    out = cv2.decomposeProjectionMatrix(cam_P[:3, :])
    cam_intrinsic = out[0]
    cam_R = out[1]
    cam_center = out[2]
    cam_center = (cam_center[:3] / cam_center[3])[:, 0]

    cam_T = -cam_R @ cam_center
    cam_extrinsic = np.eye(4)
    cam_extrinsic[:3, :3] = cam_R
    cam_extrinsic[:3, 3] = cam_T

    original_cam_extrinsic = seq_file['cam_poses'][idx + start_idx]

    if estimated:

        est_K = np.eye(4)
        est_K[:3, :3] = cam_intrinsic
        scaled_mesh.apply_transform(np.linalg.inv(original_cam_extrinsic) @ (np.linalg.inv(original_K) @ est_K) @ cam_extrinsic) # 
    else:
        scaled_mesh.apply_transform(np.linalg.inv(original_cam_extrinsic) @ cam_extrinsic)
    _ = scaled_mesh.export(os.path.join(save_dir, os.path.basename(mesh_path)))
    
