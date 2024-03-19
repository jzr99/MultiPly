import cv2
import glob
import json
import numpy as np
import os
import torch
import tqdm
import pickle as pkl
import shutil
import trimesh
from aitviewer.configuration import CONFIG as C

from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.models.smpl import SMPLLayer
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.renderables.billboard import Billboard
from aitviewer.viewer import Viewer
from aitviewer.headless import HeadlessRenderer
from scipy.spatial.transform import Rotation as R
from aitviewer.renderables.meshes import VariableTopologyMeshes
from aitviewer.renderables.meshes import Meshes
from aitviewer.scene.material import Material

# from iphone_utils import load_capture_root


def main_without_extrinsics(v2a_root, v2a_output_root, visualize_smpl, visualize_mesh):
    v = Viewer()
    v.playback_fps = 25.0
    if visualize_smpl:
        CONTACT_COLORS = [[[0.412,0.663,1.0,1.0], [1.0, 0.412, 0.514, 1.0]], [[1.0,0.749,0.412,1.0], [1.0, 0.412, 0.514, 1.0]]] 
        
        # load one frames
        frame_id = 131
        npy_path_list = sorted(glob.glob(os.path.join(v2a_output_root, 'test_dumped_smpl/*')))
        body_pose_list = []
        betas_list = []
        transl_list = []
        for i in range(len(npy_path_list)):
            npy_path = npy_path_list[i]
            print(f'Loading npy {npy_path}')
            npy_pose_path = os.path.join(npy_path, 'poses.npy')
            npy_betas_path = os.path.join(npy_path, 'mean_shape.npy')
            npy_transl_path = os.path.join(npy_path, 'normalize_trans.npy')
            # load
            body_pose = np.load(npy_pose_path)
            betas = np.load(npy_betas_path)
            transl = np.load(npy_transl_path)
            body_pose_list.append(body_pose[frame_id])
            betas_list.append(betas)
            transl_list.append(transl[frame_id])
        body_pose = np.stack(body_pose_list, axis=0)
        betas = np.stack(betas_list, axis=0)
        transl = np.stack(transl_list, axis=0)
        total_num = body_pose.shape[0]
        extrinsics_frame = extrinsics[frame_id]
        # repeat extrinsics_frame
        extrinsics_repeat = np.repeat(extrinsics_frame[np.newaxis], total_num, axis=0)
        image_files_frame = image_files[frame_id]
        image_files_repeat = [image_files_frame for i in range(total_num)]
        cameras = OpenCVCamera(intrinsics, extrinsics_repeat, cols=cols, rows=rows, viewer=v)
        bb = Billboard.from_camera_and_distance(cameras, 10.0, cols, rows, image_files_repeat)
        v.scene.add(cameras, bb)


        gender_list = np.load(f'{v2a_root}/gender.npy')

        smpl_layer = SMPLLayer(model_type='smpl', gender=gender_list[0])
        if len(betas.shape) == 3:
            smpl_seq = SMPLSequence(body_pose[:, 0, 3:],
                                smpl_layer,
                                poses_root=body_pose[:, 0, :3],
                                betas=betas[:, 0],
                                trans=transl[:, 0])
        else:
            smpl_seq = SMPLSequence(body_pose[:, 0, 3:],
                                    smpl_layer,
                                    poses_root=body_pose[:, 0, :3],
                                    betas=betas[0].reshape((1, -1)),
                                    trans=transl[:, 0])
        smpl_seq.mesh_seq.vertex_colors = np.array(CONTACT_COLORS[0])[np.zeros(6890,dtype=np.int32)][np.newaxis,...].repeat(transl.shape[0],axis=0)
        smpl_seq.name = "smpl" + str(0)
        smpl_seq.mesh_seq.material.diffuse = 1.0
        smpl_seq.mesh_seq.material.ambient = 0.1
        v.scene.add(smpl_seq)

        smpl_layer_1 = SMPLLayer(model_type='smpl', gender=gender_list[1])
        if len(betas.shape) == 3:
            # import pdb; pdb.set_trace()
            smpl_seq_1 = SMPLSequence(body_pose[:, 1, 3:],
                                smpl_layer_1,
                                poses_root=body_pose[:, 1, :3],
                                betas=betas[:, 1],
                                trans=transl[:, 1])
        else:
            smpl_seq_1 = SMPLSequence(body_pose[:, 1, 3:],
                                    smpl_layer_1,
                                    poses_root=body_pose[:, 1, :3],
                                    betas=betas[1].reshape((1, -1)),
                                    trans=transl[:, 1])
        smpl_seq_1.mesh_seq.vertex_colors = np.array(CONTACT_COLORS[1])[np.zeros(6890,dtype=np.int32)][np.newaxis,...].repeat(transl.shape[0],axis=0)
        smpl_seq_1.name = "smpl" + str(1)
        smpl_seq_1.mesh_seq.material.diffuse = 1.0
        smpl_seq_1.mesh_seq.material.ambient = 0.1
        v.scene.add(smpl_seq_1)

    if visualize_mesh:
        camPs = np.load(os.path.join(v2a_root, 'cameras.npz'))
        norm_camPs = np.load(os.path.join(v2a_root, 'cameras_normalize.npz'))
        scale = norm_camPs['scale_mat_0'].astype(np.float32)[0,0]
        image_files = sorted(glob.glob(os.path.join(v2a_root, "image/*.png")))
        print("number of image ",len(image_files))

        COLORS = [[0.412,0.663,1.0,1.0], [1.0,0.749,0.412,1.0],[0.412,1.0,0.663,1.0],[0.412,0.412,0.663,1.0],[0.412,0.0,0.0,1.0],[0.0,0.0,0.663,1.0],[0.0,0.412,0.0,1.0],[1.0,0.0,0.0,1.0]]
        mean_list = []
        number_person = len(sorted(glob.glob(os.path.join(f'{v2a_output_root}/test_mesh_res4/*'))))
        print("number_person ",number_person)
        extrinsics = []
        intrinsics = None
        for p in range(number_person):
            vertices = []
            faces = []
            vertex_normals = []
            uvs = []
            texture_paths = []
            num_frame = len(sorted(glob.glob(os.path.join(f'{v2a_output_root}/test_mesh_res4/0/*')))) // 2
            print("number_frame ",num_frame)
            for idx in range(num_frame):
                mesh = trimesh.load(os.path.join(f'{v2a_output_root}/test_mesh_res4/{p}/{idx:04d}_deformed.ply'), process=False)
                if p == 0:
                    mean_list.append(mesh.vertices.mean(axis=0))
                mesh.vertices = mesh.vertices - mean_list[idx]
                vertices.append(mesh.vertices * scale)
                faces.append(mesh.faces)
                vertex_normals.append(mesh.vertex_normals)
                if p == 0:
                    out = cv2.decomposeProjectionMatrix(camPs[f'cam_{idx}'][:3, :])
                    if idx == 0:
                        intrinsics = out[0]
                    render_R = out[1]
                    cam_center = out[2]
                    cam_center = (cam_center[:3] / cam_center[3])[:, 0]
                    cam_center = cam_center - (mean_list[idx] * scale)
                    render_T = -render_R @ cam_center
                    ext = np.zeros((4, 4))
                    ext[:3, :3] = render_R
                    ext[:3, 3] = render_T
                    extrinsics.append(ext[np.newaxis])
            m=Material(color = COLORS[p], diffuse=1.0, ambient=0.0)
            meshes = VariableTopologyMeshes(vertices,
                                            faces,
                                            vertex_normals,
                                            preload=True,
                                            color = COLORS[p],
                                            name=f"mesh_{p}",
                                            material=m,
                                            )
            # meshes.color = tuple(COLORS[p])
            # meshes.name = f"mesh_{p}"
            # meshes.material.diffuse = 1.0
            # meshes.material.ambient = 0.0
            v.scene.add(meshes)
        
        image_files = image_files[:num_frame]
        extrinsics = np.concatenate(extrinsics, axis=0)[:, :3]
        img_temp = cv2.imread(image_files[0])
        cols, rows = img_temp.shape[1], img_temp.shape[0]
        cameras = OpenCVCamera(intrinsics, extrinsics, cols=cols, rows=rows, viewer=v)
        bb = Billboard.from_camera_and_distance(cameras, 10.0, cols, rows, image_files)
        v.scene.add(cameras, bb)


    v.run()


def copy_v2a_checkpoint(root, subject_id, seq_id, v2a_vanilla_root):
    capture_root = os.path.join(root, subject_id, seq_id)
    v2a_output_root = os.path.join(capture_root, "v2a_baseline")
    v2a_output_checkpoints = os.path.join(v2a_output_root, "checkpoints")
    os.makedirs(v2a_output_checkpoints, exist_ok=True)

    src = os.path.join(v2a_vanilla_root, "emdb_{}_{}".format(subject_id, seq_id))
    checkpoint_path = sorted(glob.glob(os.path.join(src, '*.ckpt')))[-1]
    shutil.copy(checkpoint_path, v2a_output_checkpoints)

    shutil.copy(os.path.join(src, "cameras.npz"), v2a_output_root)


if __name__ == '__main__':
    # set smpl path
    C.update_conf({"smplx_models": '/media/ubuntu/hdd/Motion_infilling_smoothing/SmoothNet/data/'})
    # set v2a output path
    v2a_output_root = f'./v2a_output/dance5_sam_delay_depth_loop_noshare'
    # set v2a input path
    # v2a_input_root = f'./data/dance5'
    v2a_input_root = f'./data/dance5'
    # main_without_extrinsics(v2a_input_root, v2a_output_root, visualize_smpl=False, visualize_mesh=True)
    main_without_extrinsics(v2a_input_root, v2a_output_root, visualize_smpl=False, visualize_mesh=True)