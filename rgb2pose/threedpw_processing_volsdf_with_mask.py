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
from youtube_pose_refinement import Renderer
import warnings
warnings.filterwarnings("ignore")
def transform_smpl(curr_extrinsic, target_extrinsic, smpl_pose, smpl_trans, T_hip):
    
    R_root = cv2.Rodrigues(smpl_pose[:3])[0]
    transf_global_ori = np.linalg.inv(target_extrinsic[:3,:3]) @ curr_extrinsic[:3,:3] @ R_root
    
    target_extrinsic[:3, -1] = curr_extrinsic[:3,:3] @ (smpl_trans + T_hip) + curr_extrinsic[:3, -1] - smpl_trans - target_extrinsic[:3,:3] @ T_hip 

    smpl_pose[:3] = cv2.Rodrigues(transf_global_ori)[0].reshape(3)
    smpl_trans = np.linalg.inv(target_extrinsic[:3,:3]) @ smpl_trans # we assume

    return target_extrinsic, smpl_pose, smpl_trans

def transform_smpl_remain_extrinsic(curr_extrinsic, target_extrinsic, smpl_pose, smpl_trans, T_hip):
    R_root = cv2.Rodrigues(smpl_pose[:3])[0]
    transf_global_ori = np.linalg.inv(target_extrinsic[:3, :3]) @ curr_extrinsic[:3, :3] @ R_root

    target_extrinsic[:3, -1] = curr_extrinsic[:3, :3] @ (smpl_trans + T_hip) + curr_extrinsic[:3, -1] - smpl_trans - target_extrinsic[:3, :3] @ T_hip

    smpl_pose[:3] = cv2.Rodrigues(transf_global_ori)[0].reshape(3)
    smpl_trans = np.linalg.inv(target_extrinsic[:3, :3]) @ smpl_trans  # we assume

    smpl_trans = smpl_trans + (np.linalg.inv(target_extrinsic[:3, :3]) @ target_extrinsic[:3, -1])
    target_extrinsic[:3, -1] = np.zeros_like(target_extrinsic[:3, -1])

    return target_extrinsic, smpl_pose, smpl_trans


def render_trimesh(renderer ,mesh, R, T, mode='np'):
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None, ..., :3] / 255
    renderer.set_camera(R, T)
    image = renderer.render_mesh_recon(verts, faces, colors=colors, mode=mode)[0]
    image = (255 * image).data.cpu().numpy().astype(np.uint8)

    return image


seq = 'courtyard_shakeHands_00'

dial_kernel = np.ones((20, 20),np.uint8)

img_dir = f'/cluster/project/infk/hilliges/jiangze/ROMP/global_scratch/ROMP/ROMP/dataset/3DPW/imageFiles/{seq}'
seq_dir = f'/cluster/project/infk/hilliges/jiangze/ROMP/global_scratch/ROMP/ROMP/dataset/3DPW/sequenceFiles/train/{seq}.pkl'
# mask_dir = f'/home/chen/disk2/3DPW/smpl_mask/{seq}'

save_dir = f'/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/{seq}'
if not os.path.exists(os.path.join(save_dir, 'image')):
    os.makedirs(os.path.join(save_dir, 'image'))
if not os.path.exists(os.path.join(save_dir, 'mask')):
    os.makedirs(os.path.join(save_dir, 'mask'))


resize_factor = 2

img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))

seq_file = pkl.load(open(seq_dir, 'rb'), encoding='latin1')
smpl_model_list = []
smpl_body_shape_list = []
smpl_gender_list = []
for i in range(len(seq_file['genders'])):
    gender = seq_file['genders'][i]
    smpl_body_shape_list.append(seq_file["betas"][i][:10])
    if gender == 'f':
        gender = 'female'
    elif gender == 'm':
        gender = 'male'
    smpl_gender_list.append(gender)
    if not os.path.exists(os.path.join(save_dir, 'mask', f'{i}')):
        os.makedirs(os.path.join(save_dir, 'mask', f'{i}'))
    if not os.path.exists(os.path.join(save_dir, 'init_refined_smpl', f'{i}')):
        os.makedirs(os.path.join(save_dir, 'init_refined_smpl', f'{i}'))

    smpl_model_list.append(SMPL('/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/smpl', gender=gender))
output_body_shape = np.stack(smpl_body_shape_list, axis=0)
# we use the betas from naked body not "clothed"
np.save(os.path.join(save_dir, "gender.npy"), smpl_gender_list)
cam_intrinsics = seq_file['cam_intrinsics']

K = np.eye(4)
K[:3, :3] = cam_intrinsics
K[0, 0] = K[0, 0] / resize_factor
K[1, 1] = K[1, 1] / resize_factor
K[0, 2] = K[0, 2] / resize_factor
K[1, 2] = K[1, 2] / resize_factor
output_trans = []
output_pose = []
output_normalize_shift = []
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
    start_idx = 84
    img_paths = img_paths[start_idx:]
elif seq == 'outdoors_fencing_01':
    start_idx = 546
    img_paths = img_paths[start_idx:]
elif seq == 'downtown_warmWelcome_00':
    start_idx = 315
    end_idx = 420
    img_paths = img_paths[start_idx:end_idx]
elif seq == 'courtyard_shakeHands_00':
    start_idx = 0
    end_idx = 160
    img_paths = img_paths[start_idx:end_idx]
max_human_sphere_all = 0
for idx, img_path in enumerate(tqdm(img_paths)):
    # resize image for speed-up
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img.shape[1] // resize_factor, img.shape[0] // resize_factor))
    width = img.shape[1] // resize_factor
    height = img.shape[0] // resize_factor
    input_img = img.copy()
    renderer = Renderer(img_size=[input_img.shape[0], input_img.shape[1]], cam_intrinsic=K)
    frame = int(os.path.basename(img_path)[6:11])

    # no need to dilate ground mask
    cv2.imwrite(os.path.join(save_dir, 'image/%04d.png' % idx), img)


    cam_extrinsics = seq_file['cam_poses'][idx + start_idx]

    smpl_pose_list = []
    smpl_trans_list = []
    smpl_verts_list = []

    for i, smpl_model in enumerate(smpl_model_list):
        smpl_shape = seq_file['betas'][i][:10]
        T_hip = smpl_model.get_T_hip(betas=torch.tensor(smpl_shape)[None].float()).squeeze().cpu().numpy()
        smpl_pose = seq_file['poses'][i][idx + start_idx]
        smpl_trans = seq_file['trans'][i][idx + start_idx]

        # transform the spaces such that our model space is equal to the ICON (PIFuHD) model space
        target_extrinsic = np.eye(4)
        target_extrinsic[1:3] *= -1
        target_extrinsic, smpl_pose, smpl_trans = transform_smpl_remain_extrinsic(cam_extrinsics, target_extrinsic, smpl_pose, smpl_trans, T_hip)
        smpl_output = smpl_model(betas=torch.tensor(smpl_shape)[None].float(),
                                 body_pose=torch.tensor(smpl_pose[3:])[None].float(),
                                 global_orient=torch.tensor(smpl_pose[:3])[None].float(),
                                 transl=torch.tensor(smpl_trans)[None].float())

        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
        smpl_pose_list.append(smpl_pose)
        smpl_trans_list.append(smpl_trans)
        smpl_verts_list.append(smpl_verts)


        smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
        render_R = torch.tensor(target_extrinsic[:3, :3])[None].float()
        render_T = torch.tensor(target_extrinsic[:3, 3])[None].float()
        rendered_image = render_trimesh(renderer, smpl_mesh, render_R, render_T, 'n')

        if input_img.shape[0] < input_img.shape[1]:
            rendered_image = rendered_image[abs(input_img.shape[0] - input_img.shape[1]) // 2:(input_img.shape[0] + input_img.shape[1]) // 2, ...]
        else:
            rendered_image = rendered_image[:, abs(input_img.shape[0] - input_img.shape[1]) // 2:(input_img.shape[0] + input_img.shape[1]) // 2]
        valid_mask = (rendered_image[:, :, -1] > 0)[:, :, np.newaxis]

        output_img = (rendered_image[:, :, :-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'init_refined_smpl', f"{i}", '%04d.png' % idx), output_img)
        # import ipdb;ipdb.set_trace()
        cv2.imwrite(os.path.join(save_dir, f'mask/{i}/%04d.png' % idx), valid_mask * 255)
        mask = cv2.imread(os.path.join(save_dir, f'mask/{i}/%04d.png' % idx))
        mask = cv2.dilate(mask, dial_kernel)
        cv2.imwrite(os.path.join(save_dir, f'mask/{i}/%04d.png' % idx), mask)

        # mask = cv2.imread(f"{mask_dir}/{frame:04d}.png")
        # mask = cv2.resize(mask, (mask.shape[1] // resize_factor, mask.shape[0] // resize_factor))
        # # dilate mask
        # mask = cv2.dilate(mask, dial_kernel)
        # cv2.imwrite(os.path.join(save_dir, 'mask/%04d.png' % idx), mask)

    # we need to normalize the trans for every frame due to the large global movement
    smpl_verts_all = np.concatenate(smpl_verts_list, axis=0)
    v_max = smpl_verts_all.max(axis=0)
    v_min = smpl_verts_all.min(axis=0)
    normalize_shift = -(v_max + v_min) / 2.
    smpl_trans = np.stack(smpl_trans_list, axis=0)
    smpl_pose = np.stack(smpl_pose_list, axis=0)
    trans = smpl_trans + normalize_shift.reshape(1,3)

    smpl_verts_all = smpl_verts_all + normalize_shift.reshape(1, 3)
    max_human_sphere = np.linalg.norm(smpl_verts_all, axis=1).max()
    if max_human_sphere > max_human_sphere_all:
        max_human_sphere_all = max_human_sphere
    
    target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (target_extrinsic[:3, :3] @ normalize_shift)
    # import ipdb
    # ipdb.set_trace()
    P = K @ target_extrinsic
    # for j in range(0, smpl_verts.shape[0]):
    #     padded_v = np.pad(smpl_verts[j] + normalize_shift, (0,1), 'constant', constant_values=(0,1))
    #     temp = P @ padded_v.T # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
    #     pix = (temp/temp[2])[:2]
    #     output_img = cv2.circle(img, tuple(pix.astype(np.int32)), 3, (0,255,255), -1)
    
    # cv2.imwrite('/home/chen/Desktop/test_projcam_3dpw_norm_new.png', output_img)
    # ipdb.set_trace()
    output_trans.append(trans)
    output_pose.append(smpl_pose)
    output_normalize_shift.append(normalize_shift)
    output_P[f"cam_{idx}"] = P

print("max_human_sphere_all: ", max_human_sphere_all)
np.save(os.path.join(save_dir, 'poses.npy'), np.array(output_pose))

np.save(os.path.join(save_dir, 'mean_shape.npy'), output_body_shape)
np.save(os.path.join(save_dir, 'normalize_trans.npy'), np.array(output_trans))
np.save(os.path.join(save_dir, 'normalize_shift.npy'), np.array(output_normalize_shift))
print('output_pose', np.array(output_pose).shape)
print('mean_shape', output_body_shape.shape)
print('normalize_trans', np.array(output_trans).shape)
# np.save(os.path.join(save_dir, 'cameras.npy'), np.array(output_P))
np.savez(os.path.join(save_dir, "cameras.npz"), **output_P)

    # re-project to images to debug
smpl_shape = output_body_shape[0]
smpl_pose = smpl_pose[0]
trans = trans[0]
smpl_output = smpl_model_list[0](betas=torch.tensor(smpl_shape)[None].float(),
                         body_pose=torch.tensor(smpl_pose[3:])[None].float(),
                         global_orient=torch.tensor(smpl_pose[:3])[None].float(),
                         transl=torch.tensor(trans)[None].float())
smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
_ = trimesh.Trimesh(smpl_verts).export(os.path.join(save_dir, 'test0.ply'))

for j in range(0, smpl_verts.shape[0]):
    padded_v = np.pad(smpl_verts[j], (0,1), 'constant', constant_values=(0,1))
    temp = P @ padded_v.T # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
    pix = (temp/temp[2])[:2]
    output_img = cv2.circle(img, tuple(pix.astype(np.int32)), 3, (0,255,255), -1)
ipdb.set_trace()
cv2.imwrite(os.path.join(save_dir, 'test0.png'), output_img)

