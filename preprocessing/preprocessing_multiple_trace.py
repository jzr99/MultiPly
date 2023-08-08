import numpy as np
import pickle as pkl
import torch
import trimesh
import cv2
import os
from tqdm import tqdm
import glob
import argparse
from preprocessing_utils import (smpl_to_pose, PerspectiveCamera, Renderer, render_trimesh, \
                                estimate_translation_cv2, transform_smpl)
from loss import joints_2d_loss, pose_temporal_loss, get_loss_weights
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline

def interpolate_rotations(rotations, ts_in, ts_out):
    """
    Interpolate rotations given at timestamps `ts_in` to timestamps given at `ts_out`. This performs the equivalent
    of cubic interpolation in SO(3).
    :param rotations: A numpy array of rotations of shape (F, N, 3), i.e. rotation vectors.
    :param ts_in: Timestamps corresponding to the given rotations, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    out = []
    for j in range(rotations.shape[1]):
        rs = R.from_rotvec(rotations[:, j])
        spline = RotationSpline(ts_in, rs)
        rs_interp = spline(ts_out).as_rotvec()
        out.append(rs_interp[:, np.newaxis])
    return np.concatenate(out, axis=1)

def interpolate_positions(positions, ts_in, ts_out):
    """
    Interpolate positions given at timestamps `ts_in` to timestamps given at `ts_out` with a cubic spline.
    :param positions: A numpy array of shape (F, N, 3), i.e. in angle-axis form.
    :param ts_in: Timestamps corresponding to the given positions, len(ts_in) == F
    :param ts_out: The desired output timestamps.
    :return: A numpy array of shape (len(ts_out), N, 3).
    """
    cs = CubicSpline(ts_in, positions, axis=0)
    new_positions = cs(ts_out)
    return new_positions

def interpolate(n_frames ,frame_ids, pose_list, trans_list):
    """
    Replace the frames at the given frame IDs via an interpolation of its neighbors. Only the body pose as well
    as the root pose and translation are interpolated.
    :param frame_ids: A list of frame ids to be interpolated.
    """
    ids = np.unique(frame_ids)
    all_ids = np.arange(n_frames)
    mask_avail = np.ones(n_frames, dtype=np.bool)
    mask_avail[ids] = False

    # Interpolate poses.
    # all_poses = torch.cat([self.poses_root, self.poses_body], dim=-1)
    all_poses = pose_list
    ps = np.reshape(all_poses, (n_frames, -1, 3))
    ps_interp = interpolate_rotations(ps[mask_avail], all_ids[mask_avail], ids)
    all_poses[ids] = np.array(ps_interp.reshape(len(ids), -1))
    # self.poses_root = all_poses[:, :3]
    # self.poses_body = all_poses[:, 3:]

    # Interpolate global translation.
    ts = trans_list
    ts_interp = interpolate_positions(ts[mask_avail], all_ids[mask_avail], ids)
    trans_list[ids] = np.array(ts_interp)
    return all_poses, trans_list


def transform_smpl_remain_extrinsic(curr_extrinsic, target_extrinsic, smpl_pose, smpl_trans, T_hip):
    R_root = cv2.Rodrigues(smpl_pose[:3])[0]
    transf_global_ori = np.linalg.inv(target_extrinsic[:3, :3]) @ curr_extrinsic[:3, :3] @ R_root

    target_extrinsic[:3, -1] = curr_extrinsic[:3, :3] @ (smpl_trans + T_hip) + curr_extrinsic[:3, -1] - smpl_trans - target_extrinsic[:3, :3] @ T_hip

    smpl_pose[:3] = cv2.Rodrigues(transf_global_ori)[0].reshape(3)
    smpl_trans = np.linalg.inv(target_extrinsic[:3, :3]) @ smpl_trans  # we assume

    smpl_trans = smpl_trans + (np.linalg.inv(target_extrinsic[:3, :3]) @ target_extrinsic[:3, -1])
    target_extrinsic[:3, -1] = np.zeros_like(target_extrinsic[:3, -1])

    return target_extrinsic, smpl_pose, smpl_trans

def main(args):
    max_human_sphere_all = 0
    device = torch.device("cuda:0")
    seq = args.seq
    # gender = args.gender
    DIR = './raw_data'
    img_dir = f'{DIR}/{seq}/frames'   
    # romp_file_dir = f'{DIR}/{seq}/ROMP'
    trace_file_dir = f'{DIR}/{seq}/trace'
    # img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
    img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    # romp_file_paths = sorted(glob.glob(f"{romp_file_dir}/*.npz"))
    # format: [person_id, frame_id, ...]
    trace_file_path = f"{trace_file_dir}/{seq}.npz"
    trace_output = np.load(trace_file_path, allow_pickle=True)["results"][()]
    number_person = trace_output['smpl_betas'].shape[0]

    from smplx import SMPL
    smpl_model_list = []
    for i in range(number_person):
        smpl_model_list.append(SMPL('../code/lib/smpl/smpl_model', gender="NEUTRAL").to(device))
        # smpl_model = SMPL('../code/lib/smpl/smpl_model', gender=gender).to(device)
    
    input_img = cv2.imread(img_paths[0])
    if args.source == 'custom':
        focal_length = max(input_img.shape[0], input_img.shape[1])
        cam_intrinsics = np.array([[focal_length, 0., input_img.shape[1]//2],
                                   [0., focal_length, input_img.shape[0]//2],
                                   [0., 0., 1.]])
    elif args.source == 'neuman':
        NeuMan_DIR = '' # path to NeuMan dataset
        with open(f'{NeuMan_DIR}/{seq}/sparse/cameras.txt') as f:
            lines = f.readlines()
        cam_params = lines[3].split()
        cam_intrinsics = np.array([[float(cam_params[4]), 0., float(cam_params[6])], 
                                   [0., float(cam_params[5]), float(cam_params[7])], 
                                   [0., 0., 1.]])
    elif args.source == 'deepcap':
        DeepCap_DIR = '' # path to DeepCap dataset
        with open(f'{DeepCap_DIR}/monocularCalibrationBM.calibration') as f:
            lines = f.readlines()

        cam_params = lines[5].split()
        cam_intrinsics = np.array([[float(cam_params[1]), 0., float(cam_params[3])], 
                                   [0., float(cam_params[6]), float(cam_params[7])], 
                                   [0., 0., 1.]])
    else:
        print('Please specify the source of the dataset (custom, neuman, deepcap). We will continue to update the sources in the future.')
        raise NotImplementedError
    renderer = Renderer(img_size = [input_img.shape[0], input_img.shape[1]], cam_intrinsic=cam_intrinsics)

    if args.mode == 'mask':
        if not os.path.exists(f'{DIR}/{seq}/init_mask'):
            os.makedirs(f'{DIR}/{seq}/init_mask')
        if not os.path.exists(f'{DIR}/{seq}/init_smpl_files'):
            os.makedirs(f'{DIR}/{seq}/init_smpl_files')
        if not os.path.exists(f'{DIR}/{seq}/init_smpl_image/0'):
            os.makedirs(f'{DIR}/{seq}/init_smpl_image/0')
        if not os.path.exists(f'{DIR}/{seq}/init_smpl_image/1'):
            os.makedirs(f'{DIR}/{seq}/init_smpl_image/1')
        mean_shape = []
    elif args.mode == 'refine':
        if not os.path.exists(f'{DIR}/{seq}/init_refined_smpl/0'):
            os.makedirs(f'{DIR}/{seq}/init_refined_smpl/0')
        if not os.path.exists(f'{DIR}/{seq}/init_refined_smpl/1'):
            os.makedirs(f'{DIR}/{seq}/init_refined_smpl/1')
        if not os.path.exists(f'{DIR}/{seq}/init_refined_mask/0'):
            os.makedirs(f'{DIR}/{seq}/init_refined_mask/0')
        if not os.path.exists(f'{DIR}/{seq}/init_refined_mask/1'):
            os.makedirs(f'{DIR}/{seq}/init_refined_mask/1')
        if not os.path.exists(f'{DIR}/{seq}/init_refined_smpl_files'):
            os.makedirs(f'{DIR}/{seq}/init_refined_smpl_files')
        init_smpl_dir = f'{DIR}/{seq}/init_smpl_files'
        init_smpl_paths = sorted(glob.glob(f"{init_smpl_dir}/*.pkl"))
        openpose_dir = f'{DIR}/{seq}/openpose'
        openpose_paths = sorted(glob.glob(f"{openpose_dir}/*.npy"))
        opt_num_iters=150
        weight_dict = get_loss_weights()
        cam = PerspectiveCamera(focal_length_x=torch.tensor(cam_intrinsics[0, 0], dtype=torch.float32),
                                focal_length_y=torch.tensor(cam_intrinsics[1, 1], dtype=torch.float32),
                                center=torch.tensor(cam_intrinsics[0:2, 2]).unsqueeze(0)).to(device)
        mean_shape = []
        smpl2op_mapping = torch.tensor(smpl_to_pose(model_type='smpl', use_hands=False, use_face=False,
                                            use_face_contour=False, openpose_format='coco25'), dtype=torch.long).cuda()
    elif args.mode == 'final':
        refined_smpl_dir = f'{DIR}/{seq}/init_refined_smpl_files'
        refined_smpl_mask_dir = f'{DIR}/{seq}/init_refined_mask'
        refined_smpl_paths = sorted(glob.glob(f"{refined_smpl_dir}/*.pkl"))
        refined_smpl_mask_paths = [sorted(glob.glob(f"{refined_smpl_mask_dir}/{person_i}/*.png")) for person_i in range(number_person)]
        # refined_smpl_mask_paths = sorted(glob.glob(f"{refined_smpl_mask_dir}/*.png"))

        save_dir = f'../data/{seq}'
        if not os.path.exists(os.path.join(save_dir, 'image')):
            os.makedirs(os.path.join(save_dir, 'image'))
        if not os.path.exists(os.path.join(save_dir, 'mask')):
            os.makedirs(os.path.join(save_dir, 'mask'))

        scale_factor = args.scale_factor
        smpl_shape = np.load(f'{DIR}/{seq}/mean_shape.npy')

        K = np.eye(4)
        K[:3, :3] = cam_intrinsics
        K[0, 0] = K[0, 0] / scale_factor
        K[1, 1] = K[1, 1] / scale_factor
        K[0, 2] = K[0, 2] / scale_factor
        K[1, 2] = K[1, 2] / scale_factor

        dial_kernel = np.ones((20, 20),np.uint8)

        output_trans = []
        output_pose = []
        output_P = {}

    last_j3d = None
    actor_id = 0
    # actor_order = [0, 1]
    actor_order = list(range(number_person))
    # frame 1 must have two people
    last_smpl_verts = []
    last_pj2d = []
    last_j3d = []
    last_smpl_shape = []
    last_smpl_pose = []
    last_smpl_trans = []
    last_cam_trans = []

    last_pose = [[] for _ in range(number_person)]
    cam_extrinsics = np.eye(4)
    R = torch.tensor(cam_extrinsics[:3,:3])[None].float()
    T = torch.tensor(cam_extrinsics[:3, 3])[None].float()
    if args.mode == 'refine':
        print('start interpolate smpl')
        init_pose_list = []
        init_shape_list = []
        init_trans_list = []
        for idx, img_path in enumerate(tqdm(img_paths)):
            seq_file = pkl.load(open(init_smpl_paths[idx], 'rb'))
            init_pose_list.append(seq_file['pose'])
            init_shape_list.append(seq_file['shape'])
            init_trans_list.append(seq_file['trans'])
        init_pose_list = np.array(init_pose_list)
        init_shape_list = np.array(init_shape_list)
        init_trans_list = np.array(init_trans_list)
        # interpolate_frame_list = [25, 30, 31, 32, 33, 36, 37,38,39,40,41,42,43,44,58,65,86,116,117,131,142,202,225,235]
        # interpolate_frame_list = [17, 18, 19, 21, 32, 33, 35, 46, 47, 49, 50, 51, 71, 73, 74]
        # interpolate_frame_list = [141, 142,143, 144,145,146,147,148]
        interpolate_frame_list = []
        # interpolate_frame_list = list(range(211,224))
        # # import pdb;pdb.set_trace()
        # for person_i  in range(2):
        #     init_pose_list[:, person_i] , init_trans_list[:, person_i] = interpolate(init_pose_list.shape[0],interpolate_frame_list, init_pose_list[:, person_i], init_trans_list[:, person_i])


    for idx, img_path in enumerate(tqdm(img_paths)):
        input_img = cv2.imread(img_path)
        if args.mode == 'mask':
        # if args.mode == 'mask' or args.mode == 'refine':
            # seq_file = np.load(romp_file_paths[idx], allow_pickle=True)['results'][()]
            # seq_file = trace_output[:, idx]
            seq_file = trace_output
            cur_smpl_verts = []
            cur_pj2d = []
            cur_j3d = []
            cur_smpl_shape = []
            cur_smpl_pose = []
            cur_smpl_trans = []
            cur_cam_trans = []

            opt_pose_list = []
            opt_trans_list = []
            opt_shape_list = []
            # calculate cost matrix
            # for romp len(seq_file['smpl_thetas']) could be smaller or larger than number_person
            # cost_matrix = np.zeros((number_person, len(seq_file['smpl_thetas'])))
            # for person_i in range(number_person):
            #     if idx == 0:
            #         last_j3d.append(seq_file['joints'][actor_order, idx])
            #         last_pj2d.append(seq_file['pj2d_org'][actor_order, idx])
            #         last_cam_trans.append(seq_file['cam_trans'][actor_order, idx])
            #     for i in range(len(seq_file['smpl_thetas'])):
            #         # import pdb;pdb.set_trace()
            #         # print("joints 1 mean", seq_file['joints'][0].mean(0), "joints 2 mean", seq_file['joints'][1].mean(0),)
            #         # cost_matrix[person_i, i] = np.linalg.norm(seq_file['pj2d_org'][i].mean(0) - last_pj2d[-1][person_i].mean(0, keepdims=True)) # 2D
                      # TODO this may be wrong for trace, since joints maynot need to add cam_trans
            #         cost_matrix[person_i, i] = np.linalg.norm(seq_file['cam_trans'][i, idx] + seq_file['joints'][i, idx].mean(0) - last_j3d[-1][person_i].mean(0, keepdims=True) - last_cam_trans[-1][person_i]) # 3D
            #         # cost_matrix[person_i, i] = np.linalg.norm(seq_file['joints'][i][0] - last_j3d[-1][person_i][0])
            # # Hungarian algorithm
            # # import pdb;pdb.set_trace()
            # row_ind, col_ind = linear_sum_assignment(cost_matrix)
            # # import pdb; pdb.set_trace()

            for person_i in range(number_person):
                # tracking in case of two persons or wrong ROMP detection
                if len(seq_file['smpl_thetas']) >= number_person:
                    # dist = []
                    # if idx == 0:
                    #     last_j3d.append(seq_file['joints'][actor_order])
                    # for i in range(len(seq_file['smpl_thetas'])):
                    #     dist.append(np.linalg.norm(seq_file['joints'][i].mean(0) - last_j3d[-1][person_i].mean(0, keepdims=True)))
                    # actor_id = np.argmin(dist)
                    # assert row_ind[person_i] == person_i
                    # actor_id = col_ind[person_i]
                    actor_id = person_i
                    smpl_verts = seq_file['verts'][actor_id, idx]
                    pj2d_org = seq_file['pj2d_org'][actor_id, idx]
                    joints3d = seq_file['joints'][actor_id, idx]
                    smpl_shape = seq_file['smpl_betas'][actor_id, idx][:10]
                    smpl_pose = seq_file['smpl_thetas'][actor_id, idx]
                    cam_trans = seq_file['cam_trans'][actor_id, idx]

                # undetected person
                if len(seq_file['smpl_thetas']) < number_person:
                    # in trace, undetected person should not happened
                    # TODO we need to change the trace file to a per frame file
                    assert False
                    # TODO
                    dist = []
                    for i in range(2):
                        dist.append(np.linalg.norm(seq_file['cam_trans'][0] + seq_file['joints'][0].mean(0) - last_j3d[-1][i].mean(0, keepdims=True) - last_cam_trans[-1][i])) # 3D
                        # dist.append(np.linalg.norm(seq_file['pj2d_org'][0].mean(0) - last_pj2d[-1][i].mean(0, keepdims=True))) # 2D

                    actor_id = np.argmin(dist)
                    if actor_id == person_i:
                        smpl_verts = seq_file['verts'][0]
                        pj2d_org = seq_file['pj2d_org'][0]
                        joints3d = seq_file['joints'][0]
                        smpl_shape = seq_file['smpl_betas'][0][:10]
                        smpl_pose = seq_file['smpl_thetas'][0]
                        cam_trans = seq_file['cam_trans'][0]
                    else:
                        smpl_verts = last_smpl_verts[-1][person_i]
                        pj2d_org = last_pj2d[-1][person_i]
                        joints3d = last_j3d[-1][person_i]
                        smpl_shape = last_smpl_shape[-1][person_i][:10]
                        smpl_pose = last_smpl_pose[-1][person_i]
                        cam_trans = last_cam_trans[-1][person_i]

                cur_cam_trans.append(cam_trans)
                # J3D may be wrong for trace
                cur_j3d.append(joints3d)
                cur_pj2d.append(pj2d_org)
                cur_smpl_verts.append(smpl_verts)
                # TODO
                # last_j3d = joints3d.copy()
                
                # tra_pred = estimate_translation_cv2(joints3d, pj2d_org, proj_mat=cam_intrinsics)

                # smpl_verts += tra_pred

                opt_betas = torch.tensor(smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_pose = torch.tensor(smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_cam_tran = torch.tensor(cam_trans[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_zero = torch.tensor(np.zeros((1, 3)), dtype=torch.float32, requires_grad=True, device=device)

                smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                         body_pose=opt_pose[:, 3:],
                                         global_orient=opt_pose[:, :3],
                                         transl=opt_zero)
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                smpl_joints_3d = smpl_output.joints.data.cpu().numpy().squeeze()
                tra_pred = estimate_translation_cv2(smpl_joints_3d[:24], pj2d_org[:24], proj_mat=cam_intrinsics)
                opt_trans = torch.tensor(tra_pred[None], dtype=torch.float32, requires_grad=True, device=device)
                smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                         body_pose=opt_pose[:, 3:],
                                         global_orient=opt_pose[:, :3],
                                         transl=opt_trans)
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

                cur_smpl_trans.append(tra_pred)
                cur_smpl_shape.append(smpl_shape)
                cur_smpl_pose.append(smpl_pose)

                if args.mode == 'refine':
                    openpose = np.load(openpose_paths[idx])
                    openpose_j2d = torch.tensor(openpose[person_i, :, :2][None], dtype=torch.float32, requires_grad=False, device=device)
                    openpose_conf = torch.tensor(openpose[person_i, :,  -1][None], dtype=torch.float32, requires_grad=False, device=device)

                    # smpl_shape = seq_file['smpl_betas'][actor_id][:10]
                    # smpl_pose = seq_file['smpl_thetas'][actor_id]
                    smpl_trans = tra_pred

                    opt_betas = torch.tensor(smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
                    opt_pose = torch.tensor(smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
                    opt_trans = torch.tensor(smpl_trans[None], dtype=torch.float32, requires_grad=True, device=device)

                    opt_params = [{'params': opt_betas, 'lr': 1e-3},
                                {'params': opt_pose, 'lr': 1e-3},
                                {'params': opt_trans, 'lr': 1e-3}]
                    optimizer = torch.optim.Adam(opt_params, lr=2e-3, betas=(0.9, 0.999))
                    if idx == 0:
                        last_pose[person_i].append(opt_pose.detach().clone())
                    loop = tqdm(range(opt_num_iters))
                    for it in loop:
                        optimizer.zero_grad()

                        smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                                 body_pose=opt_pose[:,3:],
                                                 global_orient=opt_pose[:,:3],
                                                 transl=opt_trans)
                        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

                        smpl_joints_2d = cam(torch.index_select(smpl_output.joints, 1, smpl2op_mapping))

                        loss = dict()
                        weight = 1
                        # if idx <= 38 or idx >= 106 or idx == 95 or idx == 104:
                        #     weight = 1
                        # else:
                        #     weight = 0

                        # elif idx > 38 and idx < 106:
                        #     weight = 0.5
                        #     torch.linspace()
                        loss['J2D_Loss'] = joints_2d_loss(openpose_j2d, smpl_joints_2d, openpose_conf) * weight
                        loss['Temporal_Loss'] = pose_temporal_loss(last_pose[person_i][0], opt_pose)
                        w_loss = dict()
                        for k in loss:
                            w_loss[k] = weight_dict[k](loss[k], it)

                        tot_loss = list(w_loss.values())
                        tot_loss = torch.stack(tot_loss).sum()
                        tot_loss.backward()
                        optimizer.step()

                        l_str = 'Iter: %d' % it
                        for k in loss:
                            l_str += ', %s: %0.4f' % (k, weight_dict[k](loss[k], it).mean().item())
                            loop.set_description(l_str)

                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model_list[person_i].faces, process=False)
                R = torch.tensor(cam_extrinsics[:3,:3])[None].float()
                T = torch.tensor(cam_extrinsics[:3, 3])[None].float()
                rendered_image = render_trimesh(renderer, smpl_mesh, R, T, 'n')
                if input_img.shape[0] < input_img.shape[1]:
                    rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...]
                else:
                    rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]
                valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]

                if args.mode == 'mask':
                    os.makedirs(f'{DIR}/{seq}/init_mask/{person_i}', exist_ok=True)
                    cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_mask/{person_i}', '%04d.png' % idx), valid_mask*255)
                    output_img = (rendered_image[:, :, :-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
                    for keypoint_idx in range(len(pj2d_org)):
                        cv2.circle(output_img, (int(pj2d_org[keypoint_idx, 0]), int(pj2d_org[keypoint_idx, 1])), 5, (0, 0, 255), -1)
                    os.makedirs(f'{DIR}/{seq}/init_smpl_image/{person_i}', exist_ok=True)
                    cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smpl_image/{person_i}', '%04d.png' % idx), output_img)
                elif args.mode == 'refine':
                    output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
                    os.makedirs(f'{DIR}/{seq}/init_refined_smpl/{person_i}', exist_ok=True)
                    cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_smpl/{person_i}', '%04d.png' % idx), output_img)
                    os.makedirs(f'{DIR}/{seq}/init_refined_mask/{person_i}', exist_ok=True)
                    cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_mask/{person_i}', '%04d.png' % idx), valid_mask*255)
                    last_pose[person_i].pop(0)
                    last_pose[person_i].append(opt_pose.detach().clone())
                    smpl_dict = {}
                    smpl_dict['pose'] = opt_pose.data.squeeze().cpu().numpy()
                    smpl_dict['trans'] = opt_trans.data.squeeze().cpu().numpy()
                    smpl_dict['shape'] = opt_betas.data.squeeze().cpu().numpy()
                    opt_pose_list.append(smpl_dict['pose'])
                    opt_trans_list.append(smpl_dict['trans'])
                    opt_shape_list.append(smpl_dict['shape'])


            if args.mode == 'refine':
                smpl_dict = {}
                smpl_dict['pose'] = np.array(opt_pose_list)
                smpl_dict['trans'] = np.array(opt_trans_list)
                smpl_dict['shape'] = np.array(opt_shape_list)
                mean_shape.append(smpl_dict['shape'])
                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_refined_smpl_files', '%04d.pkl' % idx), 'wb'))

            last_smpl_verts.append(np.stack(cur_smpl_verts, axis=0))
            if idx != 0:
                last_j3d.append(np.stack(cur_j3d, axis=0))
                last_pj2d.append(np.stack(cur_pj2d, axis=0))
                last_cam_trans.append(np.stack(cur_cam_trans, axis=0))
            last_smpl_shape.append(np.stack(cur_smpl_shape, axis=0))
            last_smpl_pose.append(np.stack(cur_smpl_pose, axis=0))
            last_smpl_trans.append(np.stack(cur_smpl_trans, axis=0))
            if args.mode == 'mask':
                # last_smpl_verts.append(np.stack(cur_smpl_verts, axis=0))
                # last_pj2d.append(np.stack(cur_pj2d, axis=0))
                # if idx != 0:
                #     last_j3d.append(np.stack(cur_j3d, axis=0))
                # last_smpl_shape.append(np.stack(cur_smpl_shape, axis=0))
                # last_smpl_pose.append(np.stack(cur_smpl_pose, axis=0))
                # last_smpl_trans.append(np.stack(cur_smpl_trans, axis=0))

                # smpl_shape = seq_file['smpl_betas'][actor_id][:10]
                # smpl_pose = seq_file['smpl_thetas'][actor_id]
                # smpl_trans = tra_pred

                # opt_betas = torch.tensor(cur_smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
                # opt_pose = torch.tensor(cur_smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
                # opt_trans = torch.tensor(cur_smpl_trans[None], dtype=torch.float32, requires_grad=True, device=device)

                smpl_dict = {}
                # smpl_dict['pose'] = opt_pose.data.squeeze().cpu().numpy()
                # smpl_dict['trans'] = opt_trans.data.squeeze().cpu().numpy()
                # smpl_dict['shape'] = opt_betas.data.squeeze().cpu().numpy()
                smpl_dict['pose'] = np.array(cur_smpl_pose)
                smpl_dict['trans'] = np.array(cur_smpl_trans)
                smpl_dict['shape'] = np.array(cur_smpl_shape)
                mean_shape.append(smpl_dict['shape'])
                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_smpl_files', '%04d.pkl' % idx), 'wb'))
        if args.mode == 'refine':
            opt_pose_list = []
            opt_trans_list = []
            opt_shape_list = []
            for person_i in range(number_person):
                smpl_shape = init_shape_list[idx, person_i]
                smpl_pose = init_pose_list[idx, person_i]
                tra_pred = init_trans_list[idx, person_i]
                opt_betas = torch.tensor(smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_pose = torch.tensor(smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
                opt_trans = torch.tensor(tra_pred[None], dtype=torch.float32, requires_grad=True, device=device)

                smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                                        body_pose=opt_pose[:, 3:],
                                                        global_orient=opt_pose[:, :3],
                                                        transl=opt_trans)
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

                if args.mode == 'refine':
                    openpose = np.load(openpose_paths[idx])
                    if idx in interpolate_frame_list:
                        openpose_j2d = 0
                        openpose_conf = 0
                    else:
                        openpose_j2d = torch.tensor(openpose[person_i, :, :2][None], dtype=torch.float32,
                                                    requires_grad=False, device=device)
                        openpose_conf = torch.tensor(openpose[person_i, :, -1][None], dtype=torch.float32,
                                                     requires_grad=False, device=device)

                    # smpl_shape = seq_file['smpl_betas'][actor_id][:10]
                    # smpl_pose = seq_file['smpl_thetas'][actor_id]
                    smpl_trans = tra_pred

                    opt_betas = torch.tensor(smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
                    opt_pose = torch.tensor(smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
                    opt_trans = torch.tensor(smpl_trans[None], dtype=torch.float32, requires_grad=True, device=device)

                    opt_params = [{'params': opt_betas, 'lr': 1e-3},
                                  {'params': opt_pose, 'lr': 1e-3},
                                  {'params': opt_trans, 'lr': 1e-3}]
                    optimizer = torch.optim.Adam(opt_params, lr=2e-3, betas=(0.9, 0.999))
                    if idx == 0:
                        last_pose[person_i].append(opt_pose.detach().clone())
                    loop = tqdm(range(opt_num_iters))
                    for it in loop:
                        optimizer.zero_grad()

                        smpl_output = smpl_model_list[person_i](betas=opt_betas,
                                                                body_pose=opt_pose[:, 3:],
                                                                global_orient=opt_pose[:, :3],
                                                                transl=opt_trans)
                        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

                        smpl_joints_2d = cam(torch.index_select(smpl_output.joints, 1, smpl2op_mapping))

                        loss = dict()
                        weight = 1
                        if idx in interpolate_frame_list:
                            weight = 0
                            loss['J2D_Loss'] = joints_2d_loss(smpl_joints_2d, smpl_joints_2d, openpose_conf) * weight
                            loss['Temporal_Loss'] = pose_temporal_loss(last_pose[person_i][0], opt_pose)
                        else:
                            weight = 1
                            loss['J2D_Loss'] = joints_2d_loss(openpose_j2d, smpl_joints_2d, openpose_conf) * weight
                            loss['Temporal_Loss'] = pose_temporal_loss(last_pose[person_i][0], opt_pose)
                        # if idx <= 38 or idx >= 106 or idx == 95 or idx == 104:
                        #     weight = 1
                        # else:
                        #     weight = 0

                        # elif idx > 38 and idx < 106:
                        #     weight = 0.5
                        #     torch.linspace()
                        # loss['J2D_Loss'] = joints_2d_loss(openpose_j2d, smpl_joints_2d, openpose_conf) * weight
                        # loss['Temporal_Loss'] = pose_temporal_loss(last_pose[person_i][0], opt_pose)
                        w_loss = dict()
                        for k in loss:
                            w_loss[k] = weight_dict[k](loss[k], it)

                        tot_loss = list(w_loss.values())
                        tot_loss = torch.stack(tot_loss).sum()
                        tot_loss.backward()
                        optimizer.step()

                        l_str = 'Iter: %d' % it
                        for k in loss:
                            l_str += ', %s: %0.4f' % (k, weight_dict[k](loss[k], it).mean().item())
                            loop.set_description(l_str)

                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model_list[person_i].faces, process=False)
                R = torch.tensor(cam_extrinsics[:3, :3])[None].float()
                T = torch.tensor(cam_extrinsics[:3, 3])[None].float()
                rendered_image = render_trimesh(renderer, smpl_mesh, R, T, 'n')
                if input_img.shape[0] < input_img.shape[1]:
                    rendered_image = rendered_image[abs(input_img.shape[0] - input_img.shape[1]) // 2:(input_img.shape[
                                                                                                           0] +
                                                                                                       input_img.shape[
                                                                                                           1]) // 2,
                                     ...]
                else:
                    rendered_image = rendered_image[:, abs(input_img.shape[0] - input_img.shape[1]) // 2:(
                                                                                                                     input_img.shape[
                                                                                                                         0] +
                                                                                                                     input_img.shape[
                                                                                                                         1]) // 2]
                valid_mask = (rendered_image[:, :, -1] > 0)[:, :, np.newaxis]
                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model_list[person_i].faces, process=False)
                R = torch.tensor(cam_extrinsics[:3, :3])[None].float()
                T = torch.tensor(cam_extrinsics[:3, 3])[None].float()
                rendered_image = render_trimesh(renderer, smpl_mesh, R, T, 'n')
                if input_img.shape[0] < input_img.shape[1]:
                    rendered_image = rendered_image[abs(input_img.shape[0] - input_img.shape[1]) // 2:(input_img.shape[
                                                                                                           0] +
                                                                                                       input_img.shape[
                                                                                                           1]) // 2,
                                     ...]
                else:
                    rendered_image = rendered_image[:, abs(input_img.shape[0] - input_img.shape[1]) // 2:(
                                                                                                                     input_img.shape[
                                                                                                                         0] +
                                                                                                                     input_img.shape[
                                                                                                                         1]) // 2]
                valid_mask = (rendered_image[:, :, -1] > 0)[:, :, np.newaxis]

                if args.mode == 'refine':
                    output_img = (rendered_image[:, :, :-1] * valid_mask + input_img * (1 - valid_mask)).astype(
                        np.uint8)
                    os.makedirs(os.path.join(f'{DIR}/{seq}/init_refined_smpl/{person_i}'), exist_ok=True)
                    cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_smpl/{person_i}', '%04d.png' % idx), output_img)
                    os.makedirs(os.path.join(f'{DIR}/{seq}/init_refined_mask/{person_i}'), exist_ok=True)
                    cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_mask/{person_i}', '%04d.png' % idx),
                                valid_mask * 255)
                    last_pose[person_i].pop(0)
                    last_pose[person_i].append(opt_pose.detach().clone())
                    smpl_dict = {}
                    smpl_dict['pose'] = opt_pose.data.squeeze().cpu().numpy()
                    smpl_dict['trans'] = opt_trans.data.squeeze().cpu().numpy()
                    smpl_dict['shape'] = opt_betas.data.squeeze().cpu().numpy()
                    opt_pose_list.append(smpl_dict['pose'])
                    opt_trans_list.append(smpl_dict['trans'])
                    opt_shape_list.append(smpl_dict['shape'])

            if args.mode == 'refine':
                smpl_dict = {}
                smpl_dict['pose'] = np.array(opt_pose_list)
                smpl_dict['trans'] = np.array(opt_trans_list)
                smpl_dict['shape'] = np.array(opt_shape_list)
                mean_shape.append(smpl_dict['shape'])
                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_refined_smpl_files', '%04d.pkl' % idx), 'wb'))

        elif args.mode == 'final':
            input_img = cv2.resize(input_img, (input_img.shape[1] // scale_factor, input_img.shape[0] // scale_factor))
            seq_file = pkl.load(open(refined_smpl_paths[idx], 'rb'))
            cv2.imwrite(os.path.join(save_dir, 'image/%04d.png' % idx), input_img)
            smpl_pose_list = []
            smpl_trans_list = []
            smpl_verts_list = []

            for i, smpl_model in enumerate(smpl_model_list):
                mask = cv2.imread(refined_smpl_mask_paths[i][idx])
                mask = cv2.resize(mask, (mask.shape[1] // scale_factor, mask.shape[0] // scale_factor))

                # dilate mask to obtain a coarse bbox
                mask = cv2.dilate(mask, dial_kernel)
                os.makedirs(os.path.join(save_dir, f'mask/{i}'), exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f'mask/{i}/%04d.png' % idx), mask)

                smpl_pose = seq_file['pose'][i]
                smpl_trans = seq_file['trans'][i]

                # transform the spaces such that our camera has the same orientation as the OpenGL camera
                target_extrinsic = np.eye(4)
                target_extrinsic[1:3] *= -1
                T_hip = smpl_model.get_T_hip(betas=torch.tensor(smpl_shape[i])[None].float().to(device)).squeeze().cpu().numpy()
                target_extrinsic, smpl_pose, smpl_trans = transform_smpl_remain_extrinsic(cam_extrinsics, target_extrinsic, smpl_pose, smpl_trans, T_hip)
                smpl_output = smpl_model(betas=torch.tensor(smpl_shape[i])[None].float().to(device),
                                         body_pose=torch.tensor(smpl_pose[3:])[None].float().to(device),
                                         global_orient=torch.tensor(smpl_pose[:3])[None].float().to(device),
                                         transl=torch.tensor(smpl_trans)[None].float().to(device))
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                smpl_pose_list.append(smpl_pose)
                smpl_trans_list.append(smpl_trans)
                smpl_verts_list.append(smpl_verts)

            # we need to center the human for every frame due to the potentially large global movement
            # v_max = smpl_verts.max(axis=0)
            # v_min = smpl_verts.min(axis=0)
            # normalize_shift = -(v_max + v_min) / 2.
            #
            # trans = smpl_trans + normalize_shift
            smpl_verts_all = np.concatenate(smpl_verts_list, axis=0)
            v_max = smpl_verts_all.max(axis=0)
            v_min = smpl_verts_all.min(axis=0)
            normalize_shift = -(v_max + v_min) / 2.
            smpl_trans = np.stack(smpl_trans_list, axis=0)
            smpl_pose = np.stack(smpl_pose_list, axis=0)
            trans = smpl_trans + normalize_shift.reshape(1, 3)

            smpl_verts_all = smpl_verts_all + normalize_shift.reshape(1, 3)
            max_human_sphere = np.linalg.norm(smpl_verts_all, axis=1).max()
            if max_human_sphere > max_human_sphere_all:
                max_human_sphere_all = max_human_sphere
            
            target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (target_extrinsic[:3, :3] @ normalize_shift)

            P = K @ target_extrinsic
            output_trans.append(trans)
            output_pose.append(smpl_pose)
            output_P[f"cam_{idx}"] = P

    # if args.mode == 'mask':
    #     mean_shape = np.array(mean_shape)
    #     np.save(f'{DIR}/{seq}/mean_shape.npy', mean_shape.mean(0))

    if args.mode == 'refine':
        mean_shape = np.array(mean_shape)
        np.save(f'{DIR}/{seq}/mean_shape.npy', mean_shape.mean(0))
    if args.mode == 'final':
        # TODO a huge hack here
        np.save(os.path.join(save_dir, 'gender.npy'), np.array(['neutral' for _ in range(number_person)]))
        np.save(os.path.join(save_dir, 'poses.npy'), np.array(output_pose))
        np.save(os.path.join(save_dir, 'mean_shape.npy'), smpl_shape)
        np.save(os.path.join(save_dir, 'normalize_trans.npy'), np.array(output_trans))
        np.savez(os.path.join(save_dir, "cameras.npz"), **output_P)
        np.save(os.path.join(save_dir, "max_human_sphere.npy"), np.array(max_human_sphere_all))
        print("max_human_sphere_all: ", max_human_sphere_all)
        print('output_pose', np.array(output_pose).shape)
        print('mean_shape', smpl_shape.shape)
        print('normalize_trans', np.array(output_trans).shape)

        smpl_shape = smpl_shape[1]
        smpl_pose = smpl_pose[1]
        trans = trans[1]
        smpl_output = smpl_model_list[1](betas=torch.tensor(smpl_shape)[None].cuda().float(),
                                         body_pose=torch.tensor(smpl_pose[3:])[None].cuda().float(),
                                         global_orient=torch.tensor(smpl_pose[:3])[None].cuda().float(),
                                         transl=torch.tensor(trans)[None].cuda().float())
        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
        _ = trimesh.Trimesh(smpl_verts).export(os.path.join(save_dir, 'test1.ply'))

        for j in range(0, smpl_verts.shape[0]):
            padded_v = np.pad(smpl_verts[j], (0, 1), 'constant', constant_values=(0, 1))
            temp = P @ padded_v.T  # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
            pix = (temp / temp[2])[:2]
            output_img = cv2.circle(input_img, tuple(pix.astype(np.int32)), 3, (0, 255, 255), -1)
        cv2.imwrite(os.path.join(save_dir, 'test1.png'), output_img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing data")
    # video source
    parser.add_argument('--source', type=str, default='custom', help="custom video or dataset video")
    # sequence name
    parser.add_argument('--seq', type=str)
    # gender
    # parser.add_argument('--gender', type=str, help="gender of the actor: MALE or FEMALE")
    # mode
    parser.add_argument('--mode', type=str, help="mask mode or refine mode: mask or refine or final")
    # scale factor for the input image
    parser.add_argument('--scale_factor', type=int, default=1, help="scale factor for the input image")
    args = parser.parse_args()
    main(args)