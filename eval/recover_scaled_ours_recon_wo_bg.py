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

seq = 'outdoors_fencing_01'
gender = 'male'
if seq == 'outdoors_fencing_01':
    start_idx = 0 # 546

DIR = '/home/chen/RGB-PINA/code/outputs/ThreeDPW'
save_dir = f'{DIR}/{seq}_masked_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
mesh_paths = sorted(glob.glob(f'{DIR}/{seq}_masked_wo_disp_freeze_20_every_20_opt_pose/test_mesh/*_deformed.ply'))
gt_smpl_mesh_paths = sorted(glob.glob(f'/home/chen/disk2/3DPW_GT/{seq}/smpl_mesh/*.obj'))
cam_path = f'/home/chen/RGB-PINA/data/{seq}/cameras_normalize.npz'
cam = dict(np.load(cam_path))

seq_dir = '/home/chen/disk2/3DPW/sequenceFiles/test/outdoors_fencing_01.pkl' # f'/home/chen/disk2/3DPW/sequenceFiles/test/{seq}.pkl'
seq_file = pkl.load(open(seq_dir, 'rb'), encoding='latin1')

estimated = True
resize_factor = 2
import torch
checkpoint_path = sorted(glob.glob(f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}_masked_wo_disp_freeze_20_every_20_opt_pose/checkpoints/*.ckpt'))[-1] 
checkpoint = torch.load(checkpoint_path)

betas = torch.tensor(np.load(os.path.join('/home/chen/RGB-PINA/data', seq, 'mean_shape.npy')))[None].float().cuda()
global_orient = torch.tensor(np.load(os.path.join('/home/chen/RGB-PINA/data', seq, 'poses.npy'))[:, :3]).float().cuda()
transl = torch.tensor(np.load(os.path.join('/home/chen/RGB-PINA/data', seq, 'normalize_trans.npy'))).float().cuda()
body_pose = torch.tensor(np.load(os.path.join('/home/chen/RGB-PINA/data', seq, 'poses.npy'))[:, 3:]).float().cuda()

smpl_model = SMPL('/home/chen/Models/smpl', gender=gender).cuda()
for idx, mesh_path in tqdm(enumerate(mesh_paths)):
    scaled_mesh = trimesh.load(mesh_path, process=False)
    scaling_factor = cam[f'scale_mat_{idx}'][0, 0]
    scaled_mesh.apply_scale(scaling_factor)


    if estimated:
        smpl_output = smpl_model(betas = betas,
                                 body_pose = body_pose[idx:idx+1],
                                 global_orient = global_orient[idx:idx+1],
                                 transl = transl[idx:idx+1])
        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
        smpl_mesh = trimesh.Trimesh(vertices=smpl_verts, faces=smpl_model.faces, process=False)
        gt_smpl_mesh = trimesh.load(gt_smpl_mesh_paths[idx], process=False)
        aligned_smpl_verts, scale, t, R = compute_similarity_transform(smpl_mesh.vertices, gt_smpl_mesh.vertices)
        
        aligned_verts = transform_mesh(scaled_mesh.vertices, scale, t, R)
        scaled_mesh = trimesh.Trimesh(aligned_verts, scaled_mesh.faces, process=False)
        # _ = smpl_mesh.export('/home/chen/Desktop/tmp_smpl.ply')
        # _ = scaled_mesh.export('/home/chen/Desktop/tmp_scaled.ply')

    else:
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
        scaled_mesh.apply_transform(np.linalg.inv(original_cam_extrinsic) @ cam_extrinsic)

    _ = scaled_mesh.export(os.path.join(save_dir, os.path.basename(mesh_path)))
    
