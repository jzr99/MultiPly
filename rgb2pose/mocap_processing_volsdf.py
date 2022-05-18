import cv2
import numpy as np
import pickle as pkl
import os
import glob
import torch
from pytorch3d.transforms import matrix_to_axis_angle
import trimesh

target_cam_ids = [4, 28, 52, 77]

smpl_file_paths = sorted(glob.glob(os.path.join('/home/chen/mocap/Juan_waving/smpl', '*.pkl')))
smpl_mesh_paths = sorted(glob.glob(os.path.join('/home/chen/mocap/Juan_waving/smpl', '*.obj')))
smpl_poses = []
smpl_trans = []
smpl_betas = []
normalize_shift_list = []
target_cameras = []

cameras = dict(np.load(os.path.join('/home/chen/mocap/Juan_waving/cameras', 'rgb_cameras.npz'))) 



for i, smpl_file_path in enumerate(smpl_file_paths):

    smpl_file = pkl.load(open(smpl_file_path, 'rb'))
    smpl_verts = trimesh.load(smpl_mesh_paths[i]).vertices
    v_max = smpl_verts.max(axis=0)
    v_min = smpl_verts.min(axis=0)
    # pose = matrix_to_axis_angle(smpl_file['full_pose'][0].detach()).reshape(-1).cpu().numpy()
    pose = np.vstack([smpl_file['global_orient'][0].detach().cpu().numpy().reshape(-1, 1), smpl_file['body_pose'][0].detach().cpu().numpy().reshape(-1, 1)])[:, 0]
    transl = smpl_file['transl'][0].detach().cpu().numpy()
    beta = smpl_file['betas'][0].detach().cpu().numpy()

    smpl_poses.append(pose)
    smpl_trans.append(transl)
    smpl_betas.append(beta)
    normalize_shift_list.append(-(v_max + v_min) / 2.)


smpl_poses = np.array(smpl_poses)
smpl_trans = np.array(smpl_trans)
smpl_betas = np.array(smpl_betas)
normalize_shift_array = np.array(normalize_shift_list)

normalize_shift = normalize_shift_array.mean(0)

smpl_trans += normalize_shift

mean_shape = smpl_betas.mean(0)

cam_dict = {}

for i in range(len(cameras['ids'])):
    cam_id = cameras['ids'][i]
    if cam_id in target_cam_ids:
        intrinsic = cameras['intrinsics'][i]
        extrinsic = cameras['extrinsics'][i]
        extrinsic[:3, -1] = extrinsic[:3, -1] - (extrinsic[:3, :3] @ normalize_shift)

        P = intrinsic @ extrinsic
        cam_dict['cam_%d' % cam_id] = P

np.save('/home/chen/RGB-PINA/data/mocap_juan_waving/poses.npy', smpl_poses)
np.save('/home/chen/RGB-PINA/data/mocap_juan_waving/normalize_trans.npy', smpl_trans)
np.save('/home/chen/RGB-PINA/data/mocap_juan_waving/mean_shape.npy', mean_shape)

np.savez('/home/chen/RGB-PINA/data/mocap_juan_waving/cameras.npz', **cam_dict)
np.save('/home/chen/RGB-PINA/data/mocap_juan_waving/normalize_shift.npy', normalize_shift)