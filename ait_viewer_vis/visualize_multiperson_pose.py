
Root_dir = "/Users/jiangzeren/PycharmProjects/V2A/RGB-PINA/data/downtown_warmWelcome_00"
# Root_dir = "/Users/jiangzeren/PycharmProjects/V2A/RGB-PINA/code/outputs/Hi4D/downtown_warmWelcome_00/joint_opt_smpl"
import skvideo
skvideo.setFFmpegPath('/usr/local/bin/')
print("FFmpeg path: {}".format(skvideo.getFFmpegPath()))
print("FFmpeg version: {}".format(skvideo.getFFmpegVersion()))
from aitviewer.configuration import CONFIG as C
C.update_conf({"export_dir": Root_dir})
from aitviewer.viewer import Viewer
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.renderables.multi_view_system import MultiViewSystem
from aitviewer.renderables.meshes import VariableTopologyMeshes

import numpy as np
import os
import glob
import tqdm
from pathlib import Path
import torch
from pytorch3d import transforms
import cv2

COLORS = [[0.412,0.663,1.0,1.0], [1.0,0.749,0.412,1.0]]
CONTACT_COLORS = [[[0.412,0.663,1.0,1.0], [1.0, 0.412, 0.514, 1.0]], [[1.0,0.749,0.412,1.0], [1.0, 0.412, 0.514, 1.0]]] 

HI4D_PATH = "/Users/jiangzeren/Downloads/eval/data"

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

def main(args):
    v = Viewer()
    v.scene.camera.dolly_zoom(-25)
    v.scene.camera.position[1] += 2
    v.scene.camera.target[1] += 1
    v.scene.origin.enabled = False
    v.run_animations = False
    v.playback_fps = 30

    

    # root = Path(HI4D_PATH) / args.pair / args.action
    print("Visualization:", args.vis)

    # render raw meshes
    if "org" in args.vis:
        org_meshes = VariableTopologyMeshes.from_directory(root / "frames_vis", name='org_meshes')
        v.scene.add(org_meshes)

    # render segmented raw meshes
    if "seg" in args.vis:
        
        seg_meshes = VariableTopologyMeshes.from_directory(root / "frames_vis", name='seg_meshes')
        seg_mask_paths = sorted(glob.glob(f"{root}/seg/mesh_seg_mask/*.npz"))
        seg_meshes.vertex_colors = [np.array(COLORS)[np.load(seg_mask_path)["vertices_mask"]] for seg_mask_path in seg_mask_paths]
        seg_meshes.show_texture = False
        seg_meshes.material.diffuse = 1.0
        seg_meshes.material.ambient = 0.0
        v.scene.add(seg_meshes)

    # render instance meshes
    if "instance" in args.vis:
        # Warining: This takes a lot of memory!
        for p in range(2):
            instance_paths = sorted(glob.glob(f"{root}/seg/instance/{p}/*.npz"))
            vertices, faces = [], []
            for instance_path in tqdm.tqdm(instance_paths):
                instance_params = np.load(instance_path)
                vertices.append(instance_params["vertices"])
                faces.append(instance_params["faces"])
            instance_meshes = VariableTopologyMeshes(vertices=np.array(vertices), faces=np.array(faces), color=tuple(COLORS[p]), name='instance'+ str(p), preload=False)
            instance_meshes.material.diffuse = 1.0
            instance_meshes.material.ambient = 0.0
            v.scene.add(instance_meshes)


    # # get camera parameters
    # cameras = dict(np.load(f'{root}/cameras/rgb_cameras.npz'))
    # i = int(np.where(cameras['ids'] == args.cams)[0])
    # gt_cam_intrinsics = cameras['intrinsics'][i]
    # gt_cam_extrinsics = cameras['extrinsics'][i]
    # print(gt_cam_extrinsics)

    # render SMPL
    if "smpl" in args.vis:
        # gender = dict(np.load(os.path.join(root, "meta.npz")))["genders"]
        for p in range(2):
            # smpl_layer = SMPLLayer(model_type="smpl", gender=gender[p])
            smpl_layer = SMPLLayer(model_type="smpl", gender='male')
            # smpl_paths = sorted(glob.glob(f"{root}/results/mono_smpl/{args.exp}/{args.cams}/*.npz"))
            #
            # poses_body, poses_root, betas, trans, colors = [], [], [], [], []
            # for smpl_path in tqdm.tqdm(smpl_paths):
            #     if smpl_path.split('/')[-1][0] == '0':
            #         smpl_params = np.load(smpl_path, allow_pickle=True)
            #         if "results" in smpl_params.keys():
            #             smpl_params = smpl_params['results'][()]
            #             if 'body_pose' not in smpl_params.keys():
            #                 import pdb; pdb.set_trace()
            #             if p == 1 and len(smpl_params["body_pose"]) == 1:
            #                 poses_body.append(poses_body[-1])
            #                 poses_root.append(poses_root[-1])
            #                 betas.append(betas[-1])
            #                 trans.append(trans[-1])
            #                 colors.append(colors[-1])
            #                 continue
            #             poses_body.append(smpl_params["body_pose"][p])
            #             global_orientation = smpl_params["global_orient"][p]
            #             aa = transforms.axis_angle_to_matrix(torch.from_numpy(global_orientation))
            #             aa = np.linalg.inv(gt_cam_extrinsics[:3, :3]) @ np.array(aa)
            #             global_orientation = transforms.matrix_to_axis_angle(torch.from_numpy(aa))
            #             # print(global_orientation @ np.linalg.inv(gt_cam_extrinsics[:3, :3].T))
            #             poses_root.append(np.array(global_orientation))
            #             betas.append(smpl_params["smpl_betas"][p])
            #             # trans.append(smpl_params["cam_trans"][p])
            #
            #             # should I calculate trans based on pevis joint?
            #             # maybe PELVIS is not at 0 position for 71 romp_correct experiment
            #             # predicts_j3ds = smpl_params["joints"][[p], :24] - smpl_params["joints"][[p], 0] # - pred["joints"][:,[0]][matching]
            #             predicts_j3ds = smpl_params["joints"][[p], :24]
            #             predicts_pj2ds = smpl_params["pj2d_org"][[p], :24]
            #             cam_trans = estimate_translation(predicts_j3ds, predicts_pj2ds, smpl_params['cam_trans'],
            #                                              proj_mats=gt_cam_intrinsics)
            #             # trans.append(np.linalg.inv(gt_cam_extrinsics[:3, :3]) @ (np.array(cam_trans).reshape(3) - gt_cam_extrinsics[:3, 3]))
            #             trans.append(np.linalg.inv(gt_cam_extrinsics[:3, :3]) @ (np.array(cam_trans).reshape(3) + smpl_params["joints"][p, 0] - gt_cam_extrinsics[:3, 3]) - smpl_params["joints"][p, 0])
            #             # trans.append((np.array(cam_trans).reshape(3) - gt_cam_extrinsics[:3, 3]) @ np.linalg.inv(gt_cam_extrinsics[:3, :3].T))
            #             # trans.append((np.linalg.inv(gt_cam_extrinsics[:3, :3].T)) @ smpl_params["cam_trans"][p] - gt_cam_extrinsics[:3, 3])
            #             colors.append(np.array(CONTACT_COLORS[p])[np.zeros(6890,dtype=np.int32)])

            # import pdb;pdb.set_trace()
            # Root_dir = "/Users/jiangzeren/PycharmProjects/V2A/RGB-PINA/data/downtown_warmWelcome_00"
            poses = np.load(os.path.join(Root_dir,'poses.npy'))
            norm_trans = np.load(os.path.join(Root_dir, 'normalize_trans.npy'))
            mean_beats = np.load(os.path.join(Root_dir, "mean_shape.npy")).reshape(1,2,10).repeat(norm_trans.shape[0], axis=0)
            smpl_seq = SMPLSequence(poses_body = poses[:,p,3:],
                                    smpl_layer = smpl_layer,
                                    poses_root = poses[:,p,:3],
                                    betas = mean_beats[:,p,:],
                                    trans = norm_trans[:,p,:])
            # smpl_seq = SMPLSequence(poses_body = np.array(poses_body),
            #                         smpl_layer = smpl_layer,
            #                         poses_root = np.array(poses_root),
            #                         betas = np.array(betas),
            #                         trans = np.array(trans))
            smpl_seq.mesh_seq.vertex_colors = np.array(CONTACT_COLORS[p])[np.zeros(6890,dtype=np.int32)][np.newaxis,...].repeat(norm_trans.shape[0],axis=0)
            smpl_seq.name = "smpl" + str(p)
            smpl_seq.mesh_seq.material.diffuse = 1.0
            smpl_seq.mesh_seq.material.ambient = 0.1
            v.scene.add(smpl_seq)

    # render cameras systems
    if "rgb" in args.vis:
        cols = 940
        rows = 1280
        camera_system = MultiViewSystem(os.path.join(root, 'cameras/rgb_cameras.npz'),
                                        os.path.join(root, 'images'), cols, rows, v)
        camera_system._billboards_enabled = True
        # camera_system.view_from_camera(0)

        v.scene.add(camera_system)

    v.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pair', type=str,)
    parser.add_argument('--action', type=str,)
    parser.add_argument('--vis', nargs='+', type=str, default=['smpl'], help='visualize type: org, seg, instance, smpl, rgb')
    parser.add_argument('--cams', type=int, default=52)
    parser.add_argument('--exp', type=str, default='romp')
    main(parser.parse_args())
