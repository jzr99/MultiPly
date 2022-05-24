import cv2
import numpy as np
import pickle as pkl
import os
import glob
import torch
from pytorch3d.transforms import matrix_to_axis_angle
import trimesh

def func1():

    target_cam_ids = [4, 28, 52, 77]

    smpl_file_paths = sorted(glob.glob(os.path.join('/home/chen/disk2/motion_capture/markus/smpl', '*.pkl')))
    smpl_mesh_paths = sorted(glob.glob(os.path.join('/home/chen/disk2/motion_capture/markus/smpl', '*.ply')))
    smpl_poses = []
    smpl_trans = []
    smpl_betas = []
    normalize_shift_list = []

    cameras = dict(np.load(os.path.join('/home/chen/disk2/motion_capture/markus/cameras', 'rgb_cameras.npz'))) 



    for i, smpl_file_path in enumerate(smpl_file_paths):

        smpl_file = pkl.load(open(smpl_file_path, 'rb'))
        smpl_verts = trimesh.load(smpl_mesh_paths[i]).vertices
        v_max = smpl_verts.max(axis=0)
        v_min = smpl_verts.min(axis=0)
        # pose = matrix_to_axis_angle(smpl_file['full_pose'][0].detach()).reshape(-1).cpu().numpy()

        pose = smpl_file['pose'] # np.vstack([smpl_file['global_orient'][0].detach().cpu().numpy().reshape(-1, 1), smpl_file['body_pose'][0].detach().cpu().numpy().reshape(-1, 1)])[:, 0]
        transl = smpl_file['trans']
        beta = smpl_file['betas']

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

    np.save('/home/chen/disk2/motion_capture/markus/poses.npy', smpl_poses)
    np.save('/home/chen/disk2/motion_capture/markus/normalize_trans.npy', smpl_trans)
    np.save('/home/chen/disk2/motion_capture/markus/mean_shape.npy', mean_shape)

    np.savez('/home/chen/disk2/motion_capture/markus/cameras.npz', **cam_dict)
    np.save('/home/chen/disk2/motion_capture/markus/normalize_shift.npy', normalize_shift)

def func2():
    from smplx import SMPL
    import json
    smpl_faces = np.load('/home/chen/IPNet/faces.npy')
    gender = 'male'
    smpl_model = SMPL('/home/chen/Models/smpl', gender=gender)
    target_cam_ids = [0, 2, 4, 7]

    normalize_shift_list = []
    image_dir = '/home/chen/disk2/motion_capture/Ernst/images/0'
    image_0_paths = sorted(os.listdir(image_dir))
    smpl_params = np.load('/home/chen/disk2/motion_capture/Ernst/smpl_params.npz')
    smpl_poses = smpl_params['pose'][0]
    smpl_trans = smpl_params['trans'][0]
    smpl_betas = smpl_params['betas'][0]
    mean_shape = smpl_betas.mean(0)

    for i in range(smpl_poses.shape[0]):
        frame = int(image_0_paths[i].split('.')[0])
        smpl_output = smpl_model(betas = torch.tensor(mean_shape)[None].float(),
                                 body_pose = torch.tensor(smpl_poses[i][3:])[None].float(),
                                 global_orient = torch.tensor(smpl_poses[i][:3])[None].float(),
                                 transl = torch.tensor(smpl_trans[i])[None].float())    
        
        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
        smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_faces, process=False)
        _ = smpl_mesh.export('/home/chen/disk2/motion_capture/Ernst/smpl/mesh-f%05d_smpl.obj' % frame)
        v_max = smpl_verts.max(axis=0)
        v_min = smpl_verts.min(axis=0)

        tmp_normalize_shift = -(v_max + v_min) / 2.

        normalize_shift_list.append(tmp_normalize_shift)
    
    normalize_shift_array = np.array(normalize_shift_list)

    normalize_shift = normalize_shift_array.mean(0)

    smpl_trans += normalize_shift

    cameras = json.load(open('/home/chen/disk2/motion_capture/Ernst/calibration.json', 'r'))

    cam_dict = {}
    for idx, cam_id in enumerate(target_cam_ids):

        cam = cameras['%d' % cam_id]
        intrinsic = np.array(cam['K'])
        extrinsic = np.array(cam['RT'])
        extrinsic[:3, -1] = extrinsic[:3, -1] - (extrinsic[:3, :3] @ normalize_shift)
        P = intrinsic @ extrinsic
        cam_dict['cam_%d' % idx] = P
    np.save('/home/chen/RGB-PINA/data/mocap_ernst/poses.npy', smpl_poses)
    np.save('/home/chen/RGB-PINA/data/mocap_ernst/normalize_trans.npy', smpl_trans)
    np.save('/home/chen/RGB-PINA/data/mocap_ernst/mean_shape.npy', mean_shape)

    np.savez('/home/chen/RGB-PINA/data/mocap_ernst/cameras.npz', **cam_dict)
    np.save('/home/chen/RGB-PINA/data/mocap_ernst/normalize_shift.npy', normalize_shift)
if __name__ == '__main__':
    func1()