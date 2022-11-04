import os
import cv2
import numpy as np
import glob
import trimesh
import pickle as pkl
from tqdm import tqdm
import torch
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_axis_angle
import sys
sys.path.append('/home/chen/RGB-PINA/rgb2pose')
from smplx import SMPL

from utils import rectify_pose, compute_similarity_transform, transform_mesh

seq = '00070_Dance'
# gender = 'male'
# if seq == 'outdoors_fencing_01':
    # start_idx = 0 # 546
start_idx = 0
skip = 2
DIR = f'/home/chen/disk2/ICON_new_results'
save_dir = f'{DIR}/{seq}/icon-filter/test_mesh'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
mesh_paths_raw = sorted(glob.glob(f'{DIR}/{seq}/icon-filter/obj/*_recon.obj'))
# smpl_paths = sorted(glob.glob(f'{DIR}/{seq}/icon-filter/obj/*_smpl.npy'))
smpl_mesh_paths_raw = sorted(glob.glob(f'{DIR}/{seq}/icon-filter/obj/*_smpl.obj'))
gt_smpl_mesh_paths_raw = sorted(glob.glob(f'/home/chen/disk2/RGB_PINA_MoCap/{seq}/smpl_meshes/*.obj'))

mesh_paths = mesh_paths_raw[start_idx::skip]
smpl_mesh_paths = smpl_mesh_paths_raw[start_idx::skip]
gt_smpl_mesh_paths = gt_smpl_mesh_paths_raw[start_idx + 2::skip]

if len(mesh_paths) > len(gt_smpl_mesh_paths):
    if len(mesh_paths) - len(gt_smpl_mesh_paths) == 1:
        gt_smpl_mesh_paths.append(gt_smpl_mesh_paths_raw[-1])
    else:
        import ipdb
        ipdb.set_trace()
elif len(mesh_paths) < len(gt_smpl_mesh_paths):
    if len(mesh_paths) - len(gt_smpl_mesh_paths) == -1:
        gt_smpl_mesh_paths.pop()
    else:
        import ipdb
        ipdb.set_trace()
else:
    pass

# if gender == 'f':
#     gender = 'female'
# elif gender == 'm':
#     gender = 'male'

# smpl_model = SMPL('/home/chen/Models/smpl', gender=gender).cuda()

estimated = True

assert len(smpl_mesh_paths) == len(gt_smpl_mesh_paths) == len(mesh_paths)
for idx, mesh_path in tqdm(enumerate(mesh_paths)):
    scaled_mesh = trimesh.load(mesh_path, process=False)
    smpl_mesh = trimesh.load(smpl_mesh_paths[idx], process=False)
    gt_smpl_mesh = trimesh.load(gt_smpl_mesh_paths[idx], process=False)

    # icon_smpl = np.load(smpl_paths[idx], allow_pickle=True).item()
    # betas = icon_smpl['betas']
    # body_pose = icon_smpl['pose']
    # global_orient = icon_smpl['orient']
    # trans = icon_smpl['trans']
    # pose = torch.cat([global_orient, body_pose], dim=1)
    # pose = quaternion_to_axis_angle(matrix_to_quaternion(pose))
    # # pose = rectify_pose(pose.detach().cpu().numpy().squeeze().reshape(-1))
    # # pose = torch.tensor(pose)[None].cuda()

    # smpl_output = smpl_model(betas=betas.float(),
    #                          body_pose=pose[:,3:].float(),
    #                          global_orient=pose[:, :3].float(),
    #                          transl=trans[None].float())
    
    # _ = trimesh.Trimesh(smpl_output.vertices.data.cpu().numpy().squeeze(), smpl_output.faces, process=False).export('/home/chen/disk2/ICON_new_results/outdoors_fencing_01/icon-filter/smpl_debug_no_rec.obj')


    aligned_smpl_verts, scale, t, R = compute_similarity_transform(smpl_mesh.vertices, gt_smpl_mesh.vertices)

    # _ = trimesh.Trimesh(aligned_smpl_verts, smpl_model.faces, process=False).export('/home/chen/disk2/ICON_new_results/outdoors_fencing_01/icon-filter/smpl_debug_aligned.obj')
    
    aligned_verts = transform_mesh(scaled_mesh.vertices, scale, t, R)

    scaled_mesh = trimesh.Trimesh(aligned_verts, scaled_mesh.faces, process=False)
    _ = scaled_mesh.export(os.path.join(save_dir, os.path.basename(mesh_path)))

