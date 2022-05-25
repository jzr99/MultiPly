import sys
from pathlib import Path
import numpy as np
import open3d as o3d
import pymesh
import os
import os.path as osp
import torch
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (PerspectiveCameras, RasterizationSettings,
                                MeshRasterizer)
import timeit
import glob
from tqdm import tqdm
import cv2
import trimesh


def render_scan22d(camera_path,
                   scan_folder,
                   mask_save_folder=None,
                   render_info_save_folder=None,
                   image_size=(1280, 940),
                   device='cuda:0',
                   max_faces_per_bin=80000):
    """pytorch3d rendering 3d scan into different camera views,
    input:
        camera_path: the path to camera info
        scan_folder: the path to the scan folder
        mask_save_folder: where to save the scan mask
        render_info_save_folder: where to save the pytorch3d rendering info:
            {row_idx: mask to select out vertices, N_V: #scan vertices}
    return: 
        None
    """
    # os.makedirs(mask_save_folder, exist_ok=True)
    os.makedirs(render_info_save_folder, exist_ok=True)

    # init cameras
    camera_infos = dict(np.load(camera_path))
    cameras = {}
    for i in range(len(camera_infos["ids"])):
        cam_id = camera_infos["ids"][i]
        # if cam_id not in [93]: # TODO
        #     continue
        intrinsic = camera_infos["intrinsics"][i]
        extrinsic = camera_infos["extrinsics"][i]

        focal_length = torch.tensor([(intrinsic[0, 0]).astype(np.float32),
                                     (intrinsic[1, 1]).astype(np.float32)
                                     ]).to(device).unsqueeze(0)
        principal_point = [intrinsic[0, 2], intrinsic[1, 2]]
        principal_point = torch.tensor(principal_point).float().to(
            device).unsqueeze(0)
        cam_R = torch.from_numpy(
            extrinsic[:, :3]).float().to(device).unsqueeze(0)
        cam_T = torch.from_numpy(extrinsic[:,
                                           3]).float().to(device).unsqueeze(0)
        # coordinate system adaption to PyTorch3D
        cam_R[:, :2, :] *= -1.0
        cam_T[:, :
              2] *= -1.0  # camera position in world space -> world position in camera space
        cam_R = torch.transpose(cam_R, 1, 2)  # row-major
        camera = PerspectiveCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=cam_R,
            T=cam_T,
            in_ndc=False,
            image_size=torch.tensor(image_size).unsqueeze(0),
            device=device)
        cameras[str(cam_id)] = camera

    # sort according to camera id, very important as we need to gather the rendering info of multiple views,
    # the concatenating order need to be consistent with the way we do in func. "compute_scan_cls_prob"
    cameras = dict(sorted(cameras.items()))
    for cam_id in cameras.keys():
        os.makedirs(osp.join(mask_save_folder, cam_id), exist_ok=True)

    # init rasterization related classes
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
        # bin_size=100,
        max_faces_per_bin=max_faces_per_bin)

    scan_paths = [
        osp.join(scan_folder, scan_fn) for scan_fn in os.listdir(scan_folder)
        if scan_fn.endswith('.ply') # TODO
    ]
    scan_paths.sort()

    for scan_path in tqdm(scan_paths):
        scan_fn = osp.split(scan_path)[1][6:11]
        # load raw scans
        scan = trimesh.load(scan_path, process=False) # TODO np.load(scan_path) 
        v = torch.from_numpy(scan.vertices).float().to(device) # TODO
        f = torch.from_numpy(scan.faces).float().to(device) # TODO
        scan_mesh = Meshes([v], [f]).to(device)
        packed_faces = scan_mesh.faces_packed()
        N_V = (scan_mesh.verts_packed()).shape[0]

        start = timeit.default_timer()
        # render to different camera views
        row_idx_list = []
        for cam_id, camera in cameras.items():

            rasterizer = MeshRasterizer(cameras=camera,
                                        raster_settings=raster_settings)
            # Get the output from rasterization
            fragments = rasterizer(scan_mesh)

            # pix_to_face (1, H, W, 1) -> (H, W)
            pix_to_face = fragments.pix_to_face[0, :, :, 0]

            mask = pix_to_face > -1

            mask_img = mask.int().detach().cpu().numpy()
            cv2.imwrite(
                osp.join(mask_save_folder, str(cam_id),
                         '{}.png'.format(scan_fn.zfill(6))), mask_img * 255)

        #     valid_faces_id = pix_to_face[mask]
        #     valid_verts_id = packed_faces[valid_faces_id]
        #     row_idx = torch.reshape(valid_verts_id,
        #                             (-1, 1)).squeeze().detach().cpu().numpy()
        #     row_idx_list.append(row_idx)

        # with open(
        #         osp.join(render_info_save_folder,
        #                  '{}.npz'.format(scan_fn.zfill(6))), 'wb') as f:
        #     np.savez(f, row_idx=np.concatenate(row_idx_list), N_V=N_V)

        stop = timeit.default_timer()
        print('Time: ', stop - start)

def main():
    for base_dir in [
            '/home/chen/disk2/motion_capture/otmar_waving'
    ]:

        render_scan22d(camera_path=osp.join(base_dir, 'cameras/rgb_cameras.npz'),
                       scan_folder=osp.join(base_dir, 'meshes_vis'),
                       mask_save_folder=osp.join(base_dir, 'scan_mask_undistort'),
                       render_info_save_folder=osp.join(base_dir,
                                                        'render_info_undistort'),
                                                        max_faces_per_bin=40000)


if __name__ == '__main__':
    sys.exit(main())