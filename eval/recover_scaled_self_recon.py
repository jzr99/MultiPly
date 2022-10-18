import os
import cv2
import numpy as np
import glob
import trimesh
import pickle as pkl
from utils import rectify_pose, compute_similarity_transform, transform_mesh
from tqdm import tqdm
seq = 'outdoors_fencing_01'
if seq == 'outdoors_fencing_01':
    start_idx = 0

DIR = f'/home/chen/SelfReconCode/data/{seq}/result'
save_dir = f'{DIR}/final_meshes_transformed'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

mesh_paths = sorted(glob.glob(f'{DIR}/final_meshes/*.ply'))
smpl_mesh_paths = sorted(glob.glob(f'{DIR}/smpl_meshes/*.obj'))
gt_smpl_mesh_paths = sorted(glob.glob(f'/home/chen/disk2/3DPW_GT/{seq}/smpl_mesh/*.obj'))
seq_dir = f'/home/chen/disk2/3DPW/sequenceFiles/test/{seq}.pkl'
seq_file = pkl.load(open(seq_dir, 'rb'), encoding='latin1')

cam_extrinsic = np.eye(4)

estimated = True
for idx, mesh_path in enumerate(mesh_paths):
    scaled_mesh = trimesh.load(mesh_path, process=False)
    if estimated:
        smpl_mesh = trimesh.load(smpl_mesh_paths[idx], process=False)
        gt_smpl_mesh = trimesh.load(gt_smpl_mesh_paths[idx], process=False)
        aligned_smpl_verts, scale, t, R = compute_similarity_transform(smpl_mesh.vertices, gt_smpl_mesh.vertices)

        aligned_verts = transform_mesh(scaled_mesh.vertices, scale, t, R)
        scaled_mesh = trimesh.Trimesh(aligned_verts, scaled_mesh.faces, process=False)
        _ = scaled_mesh.export(os.path.join(save_dir, os.path.basename(mesh_path)))

    else:
        original_cam_extrinsic = seq_file['cam_poses'][idx + start_idx]

        scaled_mesh.apply_transform(np.linalg.inv(original_cam_extrinsic) @ cam_extrinsic)
        _ = scaled_mesh.export(os.path.join(save_dir, os.path.basename(mesh_path)))
        