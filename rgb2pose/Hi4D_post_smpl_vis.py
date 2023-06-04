from youtube_pose_refinement import Renderer
import numpy as np
import torch
import cv2
import glob
import trimesh
import os
from tqdm import trange
from smplx import SMPL
def render_trimesh(mesh,R,T, mode='np'):
    
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255
    renderer.set_camera(R,T)
    image = renderer.render_mesh_recon(verts, faces, colors=colors, mode=mode)[0]
    image = (255*image).data.cpu().numpy().astype(np.uint8)
    
    return image
device = torch.device("cuda:0")
DATA_DIR = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data"
DIR = '/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D'
# seq = 'courtyard_shakeHands_00_no_pose_condition_interpenetration_loss'
seq = 'courtyard_shakeHands_00_loop'
# seq = 'courtyard_shakeHands_00'
data_seq = 'courtyard_shakeHands_00'
# checkpoint_version = 'epoch=0499-loss=0.03910435736179352.ckpt'
checkpoint_version = 'last-v1.ckpt'
person_id = 1
# gender = 'male'
if not os.path.exists(f'{DIR}/{seq}/joint_opt_smpl'):
    os.makedirs(f'{DIR}/{seq}/joint_opt_smpl')
# checkpoint_path = sorted(glob.glob(f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}_wo_disp_freeze_20_every_20_opt_pose/checkpoints/*.ckpt'))[-1]
checkpoint_path = f"{DIR}/{seq}/checkpoints/{checkpoint_version}"
checkpoint = torch.load(checkpoint_path)
# import ipdb;ipdb.set_trace()
betas_0 = checkpoint['state_dict']['body_model_list.0.betas.weight']
betas_1 = checkpoint['state_dict']['body_model_list.1.betas.weight']
betas = torch.cat([betas_0, betas_1], dim=0)

global_orient_0 = checkpoint['state_dict']['body_model_list.0.global_orient.weight']
global_orient_1 = checkpoint['state_dict']['body_model_list.1.global_orient.weight']
global_orient = torch.stack([global_orient_0, global_orient_1], dim=1)

transl_0 = checkpoint['state_dict']['body_model_list.0.transl.weight']
transl_1 = checkpoint['state_dict']['body_model_list.1.transl.weight']
transl = torch.stack([transl_0, transl_1], dim=1)

body_pose_0 = checkpoint['state_dict']['body_model_list.0.body_pose.weight']
body_pose_1 = checkpoint['state_dict']['body_model_list.1.body_pose.weight']
body_pose = torch.stack([body_pose_0, body_pose_1], dim=1)



np.save(os.path.join(DIR, seq, 'joint_opt_smpl', 'mean_shape.npy'), betas.detach().cpu().numpy())
np.save(os.path.join(DIR, seq, 'joint_opt_smpl', 'poses.npy'), torch.cat((global_orient, body_pose), dim=2).detach().cpu().numpy())
np.save(os.path.join(DIR, seq,'joint_opt_smpl',  'normalize_trans.npy'), transl.detach().cpu().numpy())

camPs = np.load(f'{DATA_DIR}/{data_seq}/cameras.npz')
gender_list = np.load(f'{DATA_DIR}/{data_seq}/gender.npy')
smpl_model = SMPL('/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/smpl', gender=gender_list[person_id]).to(device)

img_dir = f'{DATA_DIR}/{data_seq}/image'
img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
input_img = cv2.imread(img_paths[0])
temp_camP = camPs['cam_0']
out = cv2.decomposeProjectionMatrix(temp_camP[:3, :])
cam_intrinsics = out[0]

import ipdb;ipdb.set_trace()

renderer = Renderer(img_size = [input_img.shape[0], input_img.shape[1]], cam_intrinsic=cam_intrinsics)
keypoint_list = []

for i in trange(global_orient.shape[0]):

    input_img = cv2.imread(img_paths[i])

    out = cv2.decomposeProjectionMatrix(camPs[f'cam_{i}'][:3, :])
    render_R = out[1]
    cam_center = out[2]
    cam_center = (cam_center[:3] / cam_center[3])[:, 0]
    render_T = -render_R @ cam_center
    render_R = torch.tensor(render_R)[None].float()
    render_T = torch.tensor(render_T)[None].float()

    smpl_output = smpl_model(betas = betas[[person_id]],
                             body_pose = body_pose[i:i+1,person_id],
                             global_orient = global_orient[i:i+1,person_id],
                             transl = transl[i:i+1,person_id])

    smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
    smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
    rendered_image = render_trimesh(smpl_mesh, render_R, render_T, 'n')

    if input_img.shape[0] < input_img.shape[1]:
        rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...]
    else:
        rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]

    valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
    output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
    if not os.path.exists(f'{DIR}/{seq}/joint_opt_smpl/{person_id}'):
        os.makedirs(f'{DIR}/{seq}/joint_opt_smpl/{person_id}')
    cv2.imwrite(os.path.join(f'{DIR}/{seq}/joint_opt_smpl/{person_id}', '%04d.png' % i), output_img)

    P = camPs[f'cam_{i}']
    smpl_joints = smpl_output.joints.data.cpu().numpy().squeeze()
    # print(smpl_joints.shape)
    # exit()
    smpl_joints = smpl_joints[:27] # original smpl point + nose + eyes
    pix_list = []
    for j in range(0, smpl_joints.shape[0]):
        padded_v = np.pad(smpl_joints[j], (0, 1), 'constant', constant_values=(0, 1))
        temp = P @ padded_v.T  # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
        pix = (temp / temp[2])[:2]
        output_img = cv2.circle(input_img, tuple(pix.astype(np.int32)), 3, (0, 255, 255), -1)
        pix_list.append(pix.astype(np.int32))
    pix_tensor = np.stack(pix_list, axis=0)
    keypoint_list.append(pix_tensor)
    if not os.path.exists(f'{DIR}/{seq}/joint_opt_smpl_joint/{person_id}'):
        os.makedirs(f'{DIR}/{seq}/joint_opt_smpl_joint/{person_id}')
    cv2.imwrite(os.path.join(f'{DIR}/{seq}/joint_opt_smpl_joint/{person_id}', '%04d.png' % i), output_img)

np.save(f'{DIR}/{seq}/joint_opt_smpl_joint/{person_id}.npy', np.stack(keypoint_list, axis=0))
print(np.stack(keypoint_list, axis=0).shape)