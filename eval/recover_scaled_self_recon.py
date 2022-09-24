import os
import cv2
import numpy as np
import glob
import trimesh
import pickle as pkl
seq = 'outdoors_fencing_01'
if seq == 'outdoors_fencing_01':
    start_idx = 546

DIR = f'/home/chen/SelfReconCode/data/{seq}/result'
save_dir = f'{DIR}/final_meshes_transformed'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mesh_paths = sorted(glob.glob(f'{DIR}/final_meshes/*.ply'))

seq_dir = f'/home/chen/disk2/3DPW/sequenceFiles/test/{seq}.pkl'
seq_file = pkl.load(open(seq_dir, 'rb'), encoding='latin1')

cam_extrinsic = np.eye(4)
for idx, mesh_path in enumerate(mesh_paths):
    scaled_mesh = trimesh.load(mesh_path, process=False)
    
    original_cam_extrinsic = seq_file['cam_poses'][idx + start_idx]

    scaled_mesh.apply_transform(np.linalg.inv(original_cam_extrinsic) @ cam_extrinsic)
    _ = scaled_mesh.export(os.path.join(save_dir, os.path.basename(mesh_path)))
    