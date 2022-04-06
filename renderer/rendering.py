from renderer import Renderer
import numpy as np
import ipdb
import torch
import trimesh
import cv2
smpl_faces = np.load('/home/chen/IPNet/faces.npy')
render_image_size = 640

img_idx = 50

img = cv2.imread(f'/home/chen/snarf_idr_cg_1/data/buff_est_pose/image/image_{img_idx}.png')
extrinsics = np.load('/home/chen/snarf_idr_cg_1/data/buff_est_pose/extrinsic.npz')[f'cam_{img_idx-1}']
intrinsics = np.load('/home/chen/snarf_idr_cg_1/data/buff_est_pose/K.npz')[f'cam_{img_idx-1}']


smpl_v = trimesh.load(f'/home/chen/disk2/DSFN_dataset/datasets/buff_rgbd/pcl_seq/{img_idx:04d}.ply', process=False)

R = extrinsics[:3, :3]
T = extrinsics[:3, 3]
half_max_length = max(intrinsics[0:2, 2])
f = torch.tensor([(intrinsics[0,0]/half_max_length).astype(np.float32), (intrinsics[1,1]/half_max_length).astype(np.float32)]).unsqueeze(0)
# principal_point = [-(intrinsics[0,2]-img.shape[1]/2.)/(img.shape[1]/2.), -(intrinsics[1,2]-img.shape[0]/2.)/(img.shape[0]/2.)]
renderer = Renderer(1, render_image_size, f, ((0.0, 0.0),), torch.tensor(R)[None], torch.tensor(T)[None]).cuda()

mask, _ = renderer(torch.tensor(smpl_v.vertices)[None].cuda(), torch.tensor(smpl_faces).cuda())
import ipdb
ipdb.set_trace()

cv2.imwrite('/home/chen/Desktop/mask_debug.png', mask[0].cpu().numpy()*255)
