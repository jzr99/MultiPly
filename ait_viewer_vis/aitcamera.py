# Root_dir = "/local/home/aburzio/v2a_global/data/parkinglot"
from aitviewer.configuration import CONFIG as C
# C.update_conf({"export_dir": Root_dir})
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.multi_view_system import MultiViewSystem
from aitviewer.renderables.meshes import VariableTopologyMeshes
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.headless import HeadlessRenderer

import numpy as np
import os
import glob
import tqdm
from pathlib import Path
import torch
from aitviewer.utils.so3 import aa2rot_numpy
# from pytorch3d import transforms
# import cv2

COLORS = [[0.412,0.663,1.0,1.0], [1.0,0.749,0.412,1.0]]
CONTACT_COLORS = [[[0.412,0.663,1.0,1.0], [1.0, 0.412, 0.514, 1.0]], [[1.0,0.749,0.412,1.0], [1.0, 0.412, 0.514, 1.0]], [[0.412,1.0,0.663,1.0], [1.0, 0.412, 0.514, 1.0]], [[0.412,0.412,0.663,1.0], [1.0, 0.412, 0.514, 1.0]], [[0.412,0.0,0.0,1.0], [1.0, 0.412, 0.514, 1.0]], [[0.0,0.0,0.663,1.0], [1.0, 0.412, 0.514, 1.0]],[[0.0,0.412,0.0,1.0], [1.0, 0.412, 0.514, 1.0]],[[1.0,0.0,0.0,1.0], [1.0, 0.412, 0.514, 1.0]],[[0.0,1.0,0.0,1.0], [0.0, 0.0, 1.0, 1.0]], [[0.0,0.0,1.0,1.0], [1.0, 0.412, 0.514, 1.0]]]
def estimate_translation_cv2(joints_3d, joints_2d, proj_mat=None, cam_dist=None):
    camK = proj_mat
    ret, rvec, tvec,inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                              flags=cv2.SOLVEPNP_EPNP,reprojectionError=20,iterationsCount=100)
    if inliers is None:
        return None
    else:
        tra_pred = tvec[:,0]
        return tra_pred

def estimate_translation(joints_3d, joints_2d, org_trans, proj_mats=None, cam_dists=None):
    """Find camera translation that brings 3D joints joints_3d closest to 2D the corresponding joints_2d.
    Input:
        joints_3d: (B, K, 3) 3D joint locations
        joints: (B, K, 2) 2D joint coordinates
    Returns:
        (B, 3) camera translation vectors
    """
    trans = np.zeros((joints_3d.shape[0], 3), dtype=np.float)
    if cam_dists is None:
        cam_dists = [None for _ in range(len(joints_2d))]
    # Find the translation for each example in the batch
    for i in range(joints_3d.shape[0]):
        trans_i = estimate_translation_cv2(joints_3d[i], joints_2d[i],
                proj_mat=proj_mats, cam_dist=cam_dists[i])
        trans[i] = trans_i if trans_i is not None else org_trans[i]

    return torch.from_numpy(trans).float()

def process_idx(reorganize_idx, vids=None):
    # reorganize_idx = reorganize_idx.cpu().numpy()
    used_org_inds = np.unique(reorganize_idx)
    per_img_inds = [np.where(reorganize_idx==org_idx)[0] for org_idx in used_org_inds]

    return used_org_inds, per_img_inds

def visulize_result(renderer, outputs, imgpath, rendering_cfgs, save_dir, smpl_model_path):
    used_org_inds, per_img_inds = process_idx(outputs['reorganize_idx'])
    render_images_path = []
    for org_ind, img_inds in zip(used_org_inds, per_img_inds):
        image_path = imgpath[org_ind]
        image = cv2.imread(image_path)
        if image.shape[1]>1024:
            cv2.resize(image, (image.shape[1]//2,image.shape[0]//2))
        render_image = render_image_results(renderer, outputs, img_inds, image, rendering_cfgs, smpl_model_path)
        save_path = os.path.join(save_dir, os.path.basename(image_path))
        cv2.imwrite(save_path, render_image)
        render_images_path.append(save_path)
    return render_images_path

def main(args):
    # C.update_conf({"smplx_models": '/media/ubuntu/hdd/Motion_infilling_smoothing/SmoothNet/data/'})
    if args.headless:
        v = HeadlessRenderer(size=(1920, 1080))
    else:
        v = Viewer()
        v.scene.camera.up = np.array([0.0, -1.0, 0.0])
        v.scene.camera.dolly_zoom(-25)
        v.scene.camera.position[1] += 2
        v.scene.camera.target[1] += 1
        v.scene.origin.enabled = False
        v.run_animations = False
        v.playback_fps = 30

    print("Visualization:", args.vis)

    # render SMPL
    if "smpl" in args.vis:
        # smpl_layer = SMPLLayer(model_type="smpl", gender="male")
        # poses = np.load(os.path.join(Root_dir,'poses.npy'))
        # norm_trans = np.load(os.path.join(Root_dir, 'normalize_trans.npy'))
        # mean_beats = np.load(os.path.join(Root_dir, "mean_shape.npy")).reshape(1,10).repeat(norm_trans.shape[0], axis=0)
        # smpl_seq = SMPLSequence(poses_body = poses[:,3:],
        #                         smpl_layer = smpl_layer,
        #                         poses_root = poses[:,:3],
        #                         betas = mean_beats[:,:],
        #                         trans = norm_trans[:,:])
        # smpl_seq.mesh_seq.vertex_colors = np.array(CONTACT_COLORS[0])[np.zeros(6890,dtype=np.int32)][np.newaxis,...].repeat(norm_trans.shape[0],axis=0)
        # smpl_seq.name = "smpl" + str(0)
        # smpl_seq.mesh_seq.material.diffuse = 1.0
        # smpl_seq.mesh_seq.material.ambient = 0.1

        # cameras = np.load(os.path.join(Root_dir,'cameras.npz'))

        # import numpy as np
        # r = np.load('/media/ubuntu/hdd/ROMP/simple_romp/trace_results/trace_demo.npz', allow_pickle=True)
        trace_output = f"./trace_results/{args.seq}"
        try:
            result = np.load(trace_output + '.npz', allow_pickle=True)
        except:
            try:
                result = np.load(trace_output + '.mp4' + '.npz', allow_pickle=True)
            except:
                Exception("No trace result found!")
        # import pdb;pdb.set_trace()
        smpl_layer = SMPLLayer(model_type="smpl", gender="neutral")
        used_org_inds, per_img_person_inds = process_idx(result['outputs'][()]['reorganize_idx'])
        
        
        track_ids = result['outputs'][()]['track_ids']
        unique_id = np.unique(track_ids)
        per_img_inds = [np.where(track_ids==id)[0] for id in unique_id]

        theta_np = np.zeros((len(unique_id), len(used_org_inds), 72))
        beta_np = np.zeros((len(unique_id), len(used_org_inds), 10))
        cam_np = np.zeros((len(unique_id), len(used_org_inds), 3))
        j3d_np = np.zeros((len(unique_id), len(used_org_inds), 44, 3))
        pj2d_org_np = np.zeros((len(unique_id), len(used_org_inds), 44, 2))
        verts_np = np.zeros((len(unique_id), len(used_org_inds), 6890, 3))
        for org_ind, img_inds in zip(used_org_inds, per_img_person_inds):
            for img_ind_i in img_inds:
                track_id = track_ids[img_ind_i] -1
                # import pdb;pdb.set_trace()
                theta_np[track_id, org_ind] = result['outputs'][()]['smpl_thetas'][img_ind_i]
                beta_np[track_id, org_ind] = result['outputs'][()]['smpl_betas'][img_ind_i]
                cam_np[track_id, org_ind] = result['outputs'][()]['cam_trans'][img_ind_i]
                j3d_np[track_id, org_ind] = result['outputs'][()]['j3d'][img_ind_i]
                pj2d_org_np[track_id, org_ind] = result['outputs'][()]['pj2d_org'][img_ind_i]
                # if org_ind == 131 and track_id==1:
                #     for take in range(120,131):
                #         theta_np[1, take] = result['outputs'][()]['smpl_thetas'][img_ind_i]
                #         beta_np[1, take] = result['outputs'][()]['smpl_betas'][img_ind_i]
                #         cam_np[1, take] = result['outputs'][()]['cam_trans'][img_ind_i]
                #         j3d_np[1, take] = result['outputs'][()]['j3d'][img_ind_i]
                #         pj2d_org_np[1, take] = result['outputs'][()]['pj2d_org'][img_ind_i]
                # if org_ind == 13:
                    # theta_np[track_id, 0] = result['outputs'][()]['smpl_thetas'][img_ind_i]
                    # beta_np[track_id, 0] = result['outputs'][()]['smpl_betas'][img_ind_i]
                    # cam_np[track_id, 0] = result['outputs'][()]['cam_trans'][img_ind_i]
                    # j3d_np[track_id, 0] = result['outputs'][()]['j3d'][img_ind_i]
                    # pj2d_org_np[track_id, 0] = result['outputs'][()]['pj2d_org'][img_ind_i]
                # if track_id == 2 and org_ind == 1:
                    # theta_np[track_id, 0] = result['outputs'][()]['smpl_thetas'][img_ind_i]
                    # beta_np[track_id, 0] = result['outputs'][()]['smpl_betas'][img_ind_i]
                    # cam_np[track_id, 0] = result['outputs'][()]['cam_trans'][img_ind_i]
                    # j3d_np[track_id, 0] = result['outputs'][()]['j3d'][img_ind_i]
                    # pj2d_org_np[track_id, 0] = result['outputs'][()]['pj2d_org'][img_ind_i]
        
        
        theta_list = []
        beta_list = []
        cam_list = []
        j3d_list = []
        pj2d_org_list = []
        verts_list = []

        for i, person_id_list in enumerate(per_img_inds):

            theta = result['outputs'][()]['smpl_thetas'][person_id_list]
            beta = result['outputs'][()]['smpl_betas'][person_id_list]
            world_trans = result['outputs'][()]['world_trans'][person_id_list]
            world_rotation = result['outputs'][()]['world_global_rots'][person_id_list]
            cam_trans = result['outputs'][()]['cam_trans'][person_id_list]
            j3d = result['outputs'][()]['j3d'][person_id_list]
            pj2d_org = result['outputs'][()]['pj2d_org'][person_id_list]
            theta_list.append(theta)
            beta_list.append(beta)
            cam_list.append(cam_trans)
            j3d_list.append(j3d)
            pj2d_org_list.append(pj2d_org)

            # smpl_seq = SMPLSequence(poses_body = theta[:,3:],
            #                         smpl_layer = smpl_layer,
            #                         # poses_root = world_rotation[:,:3],
            #                         poses_root = theta[:, :3],
            #                         betas = beta[:,:],
            #                         # trans = world_trans[:,:],
            #                         trans = cam_trans[:, :])
            da_pose = np.zeros_like(theta_np[i, :,3:])
            da_pose[:, 2] = np.pi / 6
            da_pose[:, 5] = - np.pi / 6
            smpl_seq = SMPLSequence(poses_body = theta_np[i, :,3:],
                                    # poses_body = da_pose,
                                    smpl_layer = smpl_layer,
                                    # poses_root = world_rotation[:,:3],
                                    poses_root = theta_np[i, :, :3],
                                    betas = beta_np[i, :,:],
                                    # trans = world_trans[:,:],
                                    trans = cam_np[i, :, :])
            verts_list.append(smpl_seq.vertices)
            verts_np[i] = smpl_seq.vertices
            
            smpl_seq.mesh_seq.vertex_colors = np.array(CONTACT_COLORS[i])[np.zeros(6890,dtype=np.int32)][np.newaxis,...].repeat(world_trans.shape[0],axis=0)
            # smpl_seq.mesh_seq.vertex_colors = np.array(CONTACT_COLORS[i])[np.zeros(6890,dtype=np.int32)][np.newaxis,...].repeat(theta_np[i, :,3:].shape[0],axis=0)
            smpl_seq.name = "smpl" + str(i)
            smpl_seq.mesh_seq.material.diffuse = 1.0
            smpl_seq.mesh_seq.material.ambient = 0.1
            v.scene.add(smpl_seq)
        
        save_result = {
            "joints": j3d_np,
            "pj2d_org": pj2d_org_np,
            "cam_trans": cam_np,
            "smpl_thetas": theta_np,
            "smpl_betas": beta_np,
            "verts": verts_np
        }
        # np.savez(trace_output + "_reformat.npz", results=save_result)
        # uncomment
        os.makedirs(f"./raw_data/{args.seq}/trace", exist_ok=True)
        np.savez(f"./raw_data/{args.seq}/trace/{args.seq}.npz", results=save_result)
        
        # save_result = {
        #     "joints": np.stack(j3d_list, 0),
        #     "pj2d_org": np.stack(pj2d_org_list, 0),
        #     "cam_trans": np.stack(cam_list, 0),
        #     "smpl_thetas": np.stack(theta_list, 0),
        #     "smpl_betas": np.stack(beta_list, 0),
        #     "verts": np.stack(verts_list, 0)
        # }
        # np.savez(trace_output + "_reformat.npz", results=save_result)


        # dict_keys(['reorganize_idx', 'j3d', 'world_cams', 'world_trans', 'world_global_rots', 'pj2d_org', 'pj2d', 'cam_trans', 'pj2d_org_h36m17', 'joints_h36m17', 'center_confs', 'track_ids', 'smpl_thetas', 'smpl_betas'])
        
        # R = aa2rot_numpy(r['outputs'][()]["world_cams"])
        # t = r['outputs'][()]["cam_trans"]
        # K=np.array([[548,0,256], [0,548,256], [0,0,1]])
        # K_list = []
        # Rt_list = []
        # for i in range(R.shape[0]):
        #     intrinsics = np.eye(4)
        #     intrinsics[:3, :3] = K
        #     pose = np.eye(4, dtype=np.float32)
        #     pose[:3, :3] = R[i]
        #     pose[:3,3] = t[i]
        #     # P = P[:3, :4]
        #     # out = cv2.decomposeProjectionMatrix(P)
        #     # K = out[0]
        #     # R = out[1]
        #     # t = out[2]

        #     # K = K/K[2,2]
        #     # intrinsics = np.eye(4)
        #     # intrinsics[:3, :3] = K

        #     # pose = np.eye(4, dtype=np.float32)
        #     # pose[:3, :3] = R.transpose()
        #     # pose[:3,3] = (t[:3] / t[3])[:,0]

        #     # pose = np.linalg.inv(pose)

        #     K_list.append(K)
        #     Rt_list.append(pose[:3,:])

        # cv_cams = OpenCVCamera(K=np.array(K_list), Rt=np.array(Rt_list), cols=648, rows=361)
        # v.scene.add(smpl_seq, cv_cams)
    if not args.headless:
        v.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', nargs='+', type=str, default=['smpl'], help='visualize type: org, seg, instance, smpl, rgb')
    parser.add_argument('--trace_output', type=str, default='/media/ubuntu/hdd/ROMP/simple_romp/trace_results/triple02_dance02_88_clip', help='trace file')
    parser.add_argument('--seq', type=str, default="seq_name", help='seq name')
    parser.add_argument('--headless', default=False, help='headless mode', action='store_true')
    main(parser.parse_args())
