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
DIR = '/home/chen/RGB-PINA/data'
seq = 'Suarez'
gender = 'male'
if not os.path.exists(f'{DIR}/{seq}/joint_opt_smpl'):
    os.makedirs(f'{DIR}/{seq}/joint_opt_smpl')
checkpoint_path = sorted(glob.glob(f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}_wo_disp_freeze_20_every_20_opt_pose/checkpoints/*.ckpt'))[-1]
checkpoint = torch.load(checkpoint_path)

betas = checkpoint['state_dict']['body_model_params.betas.weight']
global_orient = checkpoint['state_dict']['body_model_params.global_orient.weight']
transl = checkpoint['state_dict']['body_model_params.transl.weight']
body_pose = checkpoint['state_dict']['body_model_params.body_pose.weight']


# np.save(os.path.join(DIR, seq, 'opt_mean_shape.npy'), betas.detach().cpu().numpy())
# np.save(os.path.join(DIR, seq, 'opt_poses.npy'), torch.cat((global_orient, body_pose), dim=1).detach().cpu().numpy())
# np.save(os.path.join(DIR, seq, 'opt_transl.npy'), transl.detach().cpu().numpy())
# import ipdb
# ipdb.set_trace()

camPs = np.load(f'/home/chen/RGB-PINA/data/{seq}/cameras.npz')

smpl_model = SMPL('/home/chen/Models/smpl', gender=gender).to(device)

img_dir = f'{DIR}/{seq}/image'   
img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
input_img = cv2.imread(img_paths[0])
temp_camP = camPs['cam_0']
out = cv2.decomposeProjectionMatrix(temp_camP[:3, :])
cam_intrinsics = out[0]



renderer = Renderer(img_size = [input_img.shape[0], input_img.shape[1]], cam_intrinsic=cam_intrinsics)

for i in trange(global_orient.shape[0]):

    input_img = cv2.imread(img_paths[i])

    out = cv2.decomposeProjectionMatrix(camPs[f'cam_{i}'][:3, :])
    render_R = out[1]
    cam_center = out[2]
    cam_center = (cam_center[:3] / cam_center[3])[:, 0]
    render_T = -render_R @ cam_center
    render_R = torch.tensor(render_R)[None].float()
    render_T = torch.tensor(render_T)[None].float()

    smpl_output = smpl_model(betas = betas,
                             body_pose = body_pose[i:i+1],
                             global_orient = global_orient[i:i+1],
                             transl = transl[i:i+1])

    smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
    smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
    rendered_image = render_trimesh(smpl_mesh, render_R, render_T, 'n')

    if input_img.shape[0] < input_img.shape[1]:
        rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
    else:
        rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]

    valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]  
    output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
    cv2.imwrite(os.path.join(f'{DIR}/{seq}/joint_opt_smpl', '%04d.png' % i), output_img)