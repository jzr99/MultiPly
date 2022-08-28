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

dial_kernel = np.ones((20, 20),np.uint8)

seq = 'roger'

gender = 'm'

DIR = '/home/chen/disk2/Youtube_Videos'

img_dir = f'{DIR}/{seq}/frames'
seq_dir = f'{DIR}/{seq}/init_refined_smpl_files'
mask_dir = f'{DIR}/{seq}/init_refined_mask'
# ground_mask_dir = f'/home/chen/disk2/3DPW/ground_mask/{seq}'
# normal_dir = '/home/chen/ICON/courtyard_jumpBench_01/icon-filter/normal'
# mask_dir = '/home/chen/ICON/courtyard_jumpBench_01/icon-filter/mask'
save_dir = f'/home/chen/RGB-PINA/data/{seq}'
if not os.path.exists(os.path.join(save_dir, 'image')):
    os.makedirs(os.path.join(save_dir, 'image'))
if not os.path.exists(os.path.join(save_dir, 'mask')):
    os.makedirs(os.path.join(save_dir, 'mask'))
# if not os.path.exists(os.path.join(save_dir, 'ground_mask')):
#     os.makedirs(os.path.join(save_dir, 'ground_mask'))
# if not os.path.exists(os.path.join(save_dir, 'normal')):
#     os.makedirs(os.path.join(save_dir, 'normal'))
resize_factor = 2

img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
seq_file_paths = sorted(glob.glob(f"{seq_dir}/*.pkl"))

if gender == 'f':
    gender = 'female'
elif gender == 'm':
    gender = 'male'

smpl_model = SMPL('/home/chen/Models/smpl', gender=gender)
# we use the betas from naked body not "clothed"
smpl_shape = np.load(f'{DIR}/{seq}/mean_shape.npy')
T_hip = smpl_model.get_T_hip(betas=torch.tensor(smpl_shape)[None].float()).squeeze().cpu().numpy()
focal_length = 1920 # 1280 # 995.55555556
cam_intrinsics = np.array([[focal_length, 0., 960.],[0.,focal_length, 540.],[0.,0.,1.]])
cam_extrinsics = np.eye(4)

K = np.eye(4)
K[:3, :3] = cam_intrinsics
K[0, 0] = K[0, 0] / resize_factor
K[1, 1] = K[1, 1] / resize_factor
K[0, 2] = K[0, 2] / resize_factor
K[1, 2] = K[1, 2] / resize_factor
output_trans = []
output_pose = []
output_P = {}
for idx, img_path in enumerate(tqdm(img_paths)):
    # resize image for speed-up
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img.shape[1] // resize_factor, img.shape[0] // resize_factor))

    seq_file = pkl.load(open(seq_file_paths[idx], 'rb'))

    mask = cv2.imread(mask_paths[idx]) # cv2.imread(f"{mask_dir}/{frame:04d}.png")
    mask = cv2.resize(mask, (mask.shape[1] // resize_factor, mask.shape[0] // resize_factor))
    # dilate mask
    mask = cv2.dilate(mask, dial_kernel)
    # no need to dilate ground mask
    # ground_mask = cv2.imread(f"{ground_mask_dir}/image_{frame:05d}.jpg")
    # ground_mask = cv2.resize(ground_mask, (ground_mask.shape[1] // resize_factor, ground_mask.shape[0] // resize_factor))
    # normal = cv2.imread(os.path.join(normal_dir, 'image_%05d_normal.png' % (idx)))
    # normal = cv2.resize(normal, (normal.shape[1] // resize_factor, normal.shape[0] // resize_factor))

    cv2.imwrite(os.path.join(save_dir, 'image/%04d.png' % idx), img)
    cv2.imwrite(os.path.join(save_dir, 'mask/%04d.png' % idx), mask)
    # cv2.imwrite(os.path.join(save_dir, 'ground_mask/%04d.png' % idx), ground_mask)
    # cv2.imwrite(os.path.join(save_dir, 'normal/%04d.png' % idx), normal)

    smpl_pose = seq_file['pose']
    smpl_trans = seq_file['trans']
    
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

