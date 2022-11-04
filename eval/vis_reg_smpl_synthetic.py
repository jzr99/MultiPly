import numpy as np
import trimesh
import torch
from tqdm import trange
import sys
import os
sys.path.append('/home/chen/RGB-PINA/rgb2pose')
from smplx import SMPL

seq = '00070_Dance'
smpl_output_dir = f'/home/chen/disk2/RGB_PINA_MoCap/{seq}/smpl_meshes'
if not os.path.exists(smpl_output_dir):
    os.makedirs(smpl_output_dir)

gt_smpl_params = dict(np.load(os.path.join(f'/home/chen/disk2/RGB_PINA_MoCap/{seq}', 'gt_smpl_params.npz')))
gender = gt_smpl_params['gender'].item()
smpl_model = SMPL('/home/chen/Models/smpl', gender=gender).cuda()

for idx in trange(gt_smpl_params['pose'].shape[0]):
    betas = gt_smpl_params['betas'][idx]
    pose = gt_smpl_params['pose'][idx]
    trans = gt_smpl_params['trans'][idx]
    pose[-6:] = 0.

    smpl_output = smpl_model(betas = torch.tensor(betas)[None].float().cuda(),
                             body_pose = torch.tensor(pose[3:])[None].float().cuda(),
                             global_orient = torch.tensor(pose[:3])[None].float().cuda(),
                             transl = torch.tensor(trans)[None].float().cuda())

    smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
    smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
    smpl_mesh.export(os.path.join(smpl_output_dir, f'{idx:06d}.obj'))

