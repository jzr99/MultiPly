# from youtube_pose_refinement import Renderer
import numpy as np
import torch
# import cv2
import glob
# import trimesh
import os
from tqdm import trange
# from smplx import SMPL
device = torch.device("cuda:0")
DATA_DIR = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data"
DIR = '/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D'
# seq = 'courtyard_shakeHands_00_no_pose_condition_interpenetration_loss'
# seq = 'courtyard_shakeHands_00_loop'
# seq = 'courtyard_shakeHands_00'
# seq = 'cycle_dance_SAM_ratio75_smpl_depth_map_10_samGT_personid'
# seq = 'cycle_dance_sam'
# data_seq = 'cycle_dance'
# data_seq = 'courtyard_shakeHands_00'
# checkpoint_version = 'epoch=0499-loss=0.03910435736179352.ckpt'
seq = 'Hi4D_pair17_dance17_28_sam_loop_0pose_vitpose_2_noshare'
# checkpoint_version = 'last.ckpt'
person_id = 1
# gender = 'male'
if not os.path.exists(f'{DIR}/{seq}/test_dumped_smpl'):
    os.makedirs(f'{DIR}/{seq}/test_dumped_smpl')
# checkpoint_path = sorted(glob.glob(f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}_wo_disp_freeze_20_every_20_opt_pose/checkpoints/*.ckpt'))[-1]
checkpoint_list = sorted(glob.glob(f'{DIR}/{seq}/checkpoints/epoch=*.ckpt'))
for checkpoint_path in checkpoint_list:
    # checkpoint_path = f"{DIR}/{seq}/checkpoints/{checkpoint_version}"
    epoch = int(os.path.basename(checkpoint_path)[6:10])
    print(f'epoch: {epoch}')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
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
    os.makedirs(os.path.join(DIR, seq, 'test_dumped_smpl', f'{epoch:04d}'), exist_ok=True)
    np.save(os.path.join(DIR, seq, 'test_dumped_smpl', f'{epoch:04d}', 'mean_shape.npy'), betas.detach().cpu().numpy())
    np.save(os.path.join(DIR, seq, 'test_dumped_smpl', f'{epoch:04d}', 'poses.npy'), torch.cat((global_orient, body_pose), dim=2).detach().cpu().numpy())
    np.save(os.path.join(DIR, seq, 'test_dumped_smpl', f'{epoch:04d}', 'normalize_trans.npy'), transl.detach().cpu().numpy())
