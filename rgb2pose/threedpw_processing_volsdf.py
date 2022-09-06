import numpy as np
import os
import glob
import trimesh
import cv2
import pickle as pkl
import ipdb
from tqdm import tqdm
import torch
from smplx import SMPL
import warnings
warnings.filterwarnings("ignore")
def transform_smpl(curr_extrinsic, target_extrinsic, smpl_pose, smpl_trans, T_hip):
    
    R_root = cv2.Rodrigues(smpl_pose[:3])[0]
    transf_global_ori = np.linalg.inv(target_extrinsic[:3,:3]) @ curr_extrinsic[:3,:3] @ R_root
    
    target_extrinsic[:3, -1] = curr_extrinsic[:3,:3] @ (smpl_trans + T_hip) + curr_extrinsic[:3, -1] - smpl_trans - target_extrinsic[:3,:3] @ T_hip 

    smpl_pose[:3] = cv2.Rodrigues(transf_global_ori)[0].reshape(3)
    smpl_trans = np.linalg.inv(target_extrinsic[:3,:3]) @ smpl_trans # we assume

    return target_extrinsic, smpl_pose, smpl_trans
seq = 'downtown_walkDownhill_00'

dial_kernel = np.ones((20, 20),np.uint8)

img_dir = f'/home/chen/disk2/3DPW/imageFiles/{seq}'
seq_dir = f'/home/chen/disk2/3DPW/sequenceFiles/validation/{seq}.pkl'
mask_dir = f'/home/chen/disk2/3DPW/smpl_mask/{seq}'

save_dir = f'/home/chen/RGB-PINA/data/{seq}'
if not os.path.exists(os.path.join(save_dir, 'image')):
    os.makedirs(os.path.join(save_dir, 'image'))
if not os.path.exists(os.path.join(save_dir, 'mask')):
    os.makedirs(os.path.join(save_dir, 'mask'))

resize_factor = 2

img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))

seq_file = pkl.load(open(seq_dir, 'rb'), encoding='latin1')
gender = seq_file['genders'][0]
if gender == 'f':
    gender = 'female'
elif gender == 'm':
    gender = 'male'

smpl_model = SMPL('/home/chen/Models/smpl', gender=gender)
# we use the betas from naked body not "clothed"
smpl_shape = seq_file['betas'][0][:10]
T_hip = smpl_model.get_T_hip(betas=torch.tensor(smpl_shape)[None].float()).squeeze().cpu().numpy()
cam_intrinsics = seq_file['cam_intrinsics']

K = np.eye(4)
K[:3, :3] = cam_intrinsics
K[0, 0] = K[0, 0] / resize_factor
K[1, 1] = K[1, 1] / resize_factor
K[0, 2] = K[0, 2] / resize_factor
K[1, 2] = K[1, 2] / resize_factor
output_trans = []
output_pose = []
output_P = {}
"""
courtyard_bodyScannerMotions_00:
courtyard_jumpBench_01:
downtown_walkDownhill_00: 84-436
outdoors_fencing_01: 546-941
"""
if seq == 'courtyard_bodyScannerMotions_00':
    img_paths = img_paths #[]
elif seq == 'courtyard_jumpBench_01':
    img_paths = img_paths 
elif seq == 'downtown_walkDownhill_00':
    img_paths = img_paths[84:]
elif seq == 'outdoors_fencing_01':
    img_paths = img_paths[546:]
for idx, img_path in enumerate(tqdm(img_paths)):
    # resize image for speed-up
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img.shape[1] // resize_factor, img.shape[0] // resize_factor))

    frame = int(os.path.basename(img_path)[6:11])
    mask = cv2.imread(f"{mask_dir}/{frame:04d}.png")
    mask = cv2.resize(mask, (mask.shape[1] // resize_factor, mask.shape[0] // resize_factor))
    # dilate mask
    mask = cv2.dilate(mask, dial_kernel)
    # no need to dilate ground mask
    cv2.imwrite(os.path.join(save_dir, 'image/%04d.png' % idx), img)
    cv2.imwrite(os.path.join(save_dir, 'mask/%04d.png' % idx), mask)
    # cv2.imwrite(os.path.join(save_dir, 'ground_mask/%04d.png' % idx), ground_mask)
    # cv2.imwrite(os.path.join(save_dir, 'normal/%04d.png' % idx), normal)
    cam_extrinsics = seq_file['cam_poses'][idx]

    smpl_pose = seq_file['poses'][0][idx]
    smpl_trans = seq_file['trans'][0][idx]
    
    # transform the spaces such that our model space is equal to the ICON (PIFuHD) model space  
    target_extrinsic = np.eye(4)
    target_extrinsic[1:3] *= -1
    target_extrinsic, smpl_pose, smpl_trans = transform_smpl(cam_extrinsics, target_extrinsic, smpl_pose, smpl_trans, T_hip)
    smpl_output = smpl_model(betas=torch.tensor(smpl_shape)[None].float(),
                             body_pose=torch.tensor(smpl_pose[3:])[None].float(),
                             global_orient=torch.tensor(smpl_pose[:3])[None].float(),
                             transl=torch.tensor(smpl_trans)[None].float())
    
    smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

    # we need to normalize the trans for every frame due to the large global movement
    v_max = smpl_verts.max(axis=0)
    v_min = smpl_verts.min(axis=0)
    normalize_shift = -(v_max + v_min) / 2.

    trans = smpl_trans + normalize_shift
    
    target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (target_extrinsic[:3, :3] @ normalize_shift)

    P = K @ target_extrinsic
    output_trans.append(trans)
    output_pose.append(smpl_pose)
    output_P[f"cam_{idx}"] = P

np.save(os.path.join(save_dir, 'poses.npy'), np.array(output_pose))
np.save(os.path.join(save_dir, 'mean_shape.npy'), smpl_shape)
np.save(os.path.join(save_dir, 'normalize_trans.npy'), np.array(output_trans))
# np.save(os.path.join(save_dir, 'cameras.npy'), np.array(output_P))
np.savez(os.path.join(save_dir, "cameras.npz"), **output_P)

    # re-project to images to debug

    # smpl_output = smpl_model(betas=torch.tensor(smpl_shape)[None].float(),
    #                          body_pose=torch.tensor(smpl_pose[3:])[None].float(),
    #                          global_orient=torch.tensor(smpl_pose[:3])[None].float(),
    #                          transl=torch.tensor(trans)[None].float())
    # smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
    # _ = trimesh.Trimesh(smpl_verts).export('/home/chen/Desktop/threedpw.ply')

    # for j in range(0, smpl_verts.shape[0]):
    #     padded_v = np.pad(smpl_verts[j], (0,1), 'constant', constant_values=(0,1))
    #     temp = P @ padded_v.T # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
    #     pix = (temp/temp[2])[:2]
    #     output_img = cv2.circle(img, tuple(pix.astype(np.int32)), 3, (0,255,255), -1)
    # ipdb.set_trace()
    # cv2.imwrite('/home/chen/Desktop/test_projcam_3dpw_norm_new.png', output_img)

