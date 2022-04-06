import os
import numpy as np
import torch

import pickle
import cv2
import glob
from smplx import SMPL
cam_intrinsic = [[915.8216 ,   0.     , 957.34607],
                 [  0.     , 915.64954, 543.62683],
                 [  0.     ,   0.     ,   1.     ]]

# cam_intrinsic = np.load('/home/chen/snarf_idr_cg_1/data/real/kinect_rgb_intrinsic.npy')
K = np.eye(4)
K[:3,:3] = cam_intrinsic

gender = 'male'
smpl_model = SMPL(model_path='/home/chen/Models/smpl', batch_size = 1, gender=gender).cuda()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# we choose the second sequence due to the T-pose rotataion
actor = 'jz_1'
seq_num = 8
smpl_files = sorted(glob.glob('/home/chen/disk2/kinect_capture_results/%s/%s_%d/pcl_seq_files/*.pkl' % (actor, actor, seq_num)))

image_files = sorted(glob.glob('/home/chen/disk2/kinect_capture_results/%s_src/%s_%d/image/*.png' % (actor, actor, seq_num)))

mask_files = sorted(glob.glob('/home/chen/disk2/kinect_capture_results/%s_src/%s_%d/matte/*.png'  % (actor, actor, seq_num)))

skip_step = 10

smpl_files = smpl_files[::skip_step]
image_files = image_files[::skip_step]
mask_files = mask_files[::skip_step]

save_dir = '/home/chen/snarf_idr_cg_1/data/%s' % actor

save_image_dir = os.path.join(save_dir, 'image')
save_mask_dir = os.path.join(save_dir, 'mask')
os.path.exists(save_image_dir) or os.makedirs(save_image_dir, exist_ok=True)
os.path.exists(save_mask_dir) or os.makedirs(save_mask_dir, exist_ok=True)
poses = []
trans = []
mean_shape = np.load(os.path.join(save_dir, 'mean_shape.npy'))
extrinsic = np.eye(4)
normalize_shift = 0.
for i, smpl_file_path in enumerate(smpl_files):
    smpl_params = pickle.load(open(smpl_file_path, 'rb'))
    image = cv2.imread(image_files[i]) 
    mask = cv2.imread(mask_files[i])

    if i == 0:
        smpl_output = smpl_model(betas=torch.tensor(mean_shape).unsqueeze(0).to(device),
                                 transl=torch.tensor(smpl_params['trans']).unsqueeze(0).to(device),
                                 global_orient=torch.tensor(smpl_params['pose'][:3]).unsqueeze(0).to(device),
                                 body_pose=torch.tensor(smpl_params['pose'][3:]).unsqueeze(0).to(device))
        verts = smpl_output.vertices.detach()[0].cpu().numpy()
        v_max = verts.max(axis=0)
        v_min = verts.min(axis=0)
        normalize_shift = -(v_max + v_min) / 2.
        
    trans.append(normalize_shift+smpl_params['trans'])
    poses.append(smpl_params['pose'])
    cv2.imwrite(os.path.join(save_image_dir, 'image_%04d.png' % i), image)
    cv2.imwrite(os.path.join(save_mask_dir, 'mask_%04d.png' % i), mask)

# in our monocular setting, the camera rotation is always a identity matrix
extrinsic[:3, -1] = -normalize_shift
import pdb
pdb.set_trace()
P = K @ extrinsic
np.save(os.path.join(save_dir, 'camera.npy'), P)
np.save(os.path.join(save_dir, 'poses.npy'), np.array(poses))
np.save(os.path.join(save_dir, 'normalize_trans.npy'), np.array(trans))
