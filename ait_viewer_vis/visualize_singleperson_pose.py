Root_dir = "/Users/jiangzeren/Downloads/reformat"
# import skvideo
# skvideo.setFFmpegPath('/usr/local/bin/')
# print("FFmpeg path: {}".format(skvideo.getFFmpegPath()))
# print("FFmpeg version: {}".format(skvideo.getFFmpegVersion()))
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
        smpl_layer = SMPLLayer(model_type="smpl", gender="male")
        poses = np.load(os.path.join(Root_dir,'poses.npy'))
        norm_trans = np.load(os.path.join(Root_dir, 'normalize_trans.npy'))
        mean_beats = np.load(os.path.join(Root_dir, "mean_shape.npy")).reshape(1,10).repeat(norm_trans.shape[0], axis=0)
        smpl_seq = SMPLSequence(poses_body = poses[:,3:],
                                smpl_layer = smpl_layer,
                                poses_root = poses[:,:3],
                                betas = mean_beats[:,:],
                                trans = norm_trans[:,:])
        smpl_seq.mesh_seq.vertex_colors = np.array(CONTACT_COLORS[0])[np.zeros(6890,dtype=np.int32)][np.newaxis,...].repeat(norm_trans.shape[0],axis=0)
        smpl_seq.name = "smpl" + str(0)
        smpl_seq.mesh_seq.material.diffuse = 1.0
        smpl_seq.mesh_seq.material.ambient = 0.1
        v.scene.add(smpl_seq)

    # render cameras systems
    # if "rgb" in args.vis:
    #     cols = 940
    #     rows = 1280
    #     camera_system = MultiViewSystem(os.path.join(root, 'cameras/rgb_cameras.npz'),
    #                                     os.path.join(root, 'images'), cols, rows, v)
    #     camera_system._billboards_enabled = True
    #
    #     v.scene.add(camera_system)

    v.run()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', nargs='+', type=str, default=['smpl'], help='visualize type: org, seg, instance, smpl, rgb')
    main(parser.parse_args())
