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
from aitviewer.scene.camera import PinholeCamera

# from iphone_utils import load_capture_root


def main_without_extrinsics(v2a_root, v2a_output_root, visualize_smpl, visualize_mesh):
    # v = Viewer(size=(480, 640))
    # v = HeadlessRenderer(size=(720, 640))
    # v = HeadlessRenderer(size=(480, 640))
    v = HeadlessRenderer(size=(1920, 1080))
    # v.scene.camera.position = [-4.456,0.142,2.985]
    # v.scene.camera.position = [-4,0,3.3]
    v.scene.camera.position = [-3.4,0,3.1]
    v.scene.camera.target =  [0.44254455, -0.11135479, -0.2573365]
    # v.playback_fps = 25.0
    
    v.scene.floor.enabled = True
    v.auto_set_floor = False
    v.auto_set_camera_target = False
    v.scene.floor.position = np.array([0, -1, 0])
    v.scene.origin.enabled = False

    # camPs = np.load(os.path.join(v2a_root, 'cameras.npz'))
    # norm_camPs = np.load(os.path.join(v2a_root, 'cameras_normalize.npz'))
    # scale = norm_camPs['scale_mat_0'].astype(np.float32)[0,0]
    # extrinsics = []
    # intrinsics = None
    # image_files = sorted(glob.glob(os.path.join(v2a_root, "image/*.png")))
    # print("number of image ",len(image_files))
    # for i in range(len(image_files)):
    #     out = cv2.decomposeProjectionMatrix(camPs[f'cam_{i}'][:3, :])
    #     if i == 0:
    #         intrinsics = out[0]

    #     render_R = out[1]
    #     cam_center = out[2]
    #     cam_center = (cam_center[:3] / cam_center[3])[:, 0]
    #     render_T = -render_R @ cam_center
    #     ext = np.zeros((4, 4))
    #     ext[:3, :3] = render_R
    #     ext[:3, 3] = render_T
    #     extrinsics.append(ext[np.newaxis])
    # extrinsics = np.concatenate(extrinsics, axis=0)[:, :3]
    # # image_files = sorted(glob.glob(os.path.join(v2a_root, "image/*.png")))
    # img_temp = cv2.imread(image_files[0])
    # cols, rows = img_temp.shape[1], img_temp.shape[0]
    # cameras = OpenCVCamera(intrinsics, extrinsics, cols=cols, rows=rows, viewer=v)

    # bb = Billboard.from_camera_and_distance(cameras, 10.0, cols, rows, image_files)
    # v.scene.add(cameras, bb)
    
    if visualize_smpl:
        CONTACT_COLORS = [[[0.412,0.663,1.0,1.0], [1.0, 0.412, 0.514, 1.0]], [[1.0,0.749,0.412,1.0], [1.0, 0.412, 0.514, 1.0]]] 
        """Extract the poses from the last checkpoint and visualize assuming unknown extrinsics."""
        # checkpoint_path = sorted(glob.glob(os.path.join(v2a_output_root, 'checkpoints/*.ckpt')))[-1]
        # print(f'Loading checkpoint {checkpoint_path}')
        # checkpoint = torch.load(checkpoint_path)

        # betas_0 = checkpoint['state_dict']['body_model_list.0.betas.weight'].cpu()
        # betas_1 = checkpoint['state_dict']['body_model_list.1.betas.weight'].cpu()
        # betas = torch.cat([betas_0, betas_1], dim=0).numpy()

        # global_orient_0 = checkpoint['state_dict']['body_model_list.0.global_orient.weight'].cpu()
        # global_orient_1 = checkpoint['state_dict']['body_model_list.1.global_orient.weight'].cpu()
        # global_orient = torch.stack([global_orient_0, global_orient_1], dim=1).numpy()

        # transl_0 = checkpoint['state_dict']['body_model_list.0.transl.weight'].cpu()
        # transl_1 = checkpoint['state_dict']['body_model_list.1.transl.weight'].cpu()
        # transl = torch.stack([transl_0, transl_1], dim=1).numpy()

        # body_pose_0 = checkpoint['state_dict']['body_model_list.0.body_pose.weight'].cpu()
        # body_pose_1 = checkpoint['state_dict']['body_model_list.1.body_pose.weight'].cpu()
        # body_pose = torch.stack([body_pose_0, body_pose_1], dim=1).numpy()
        
        
        # uncomment this to load all smpl
        # load all frames
        # npy_path = sorted(glob.glob(os.path.join(v2a_output_root, 'test_dumped_smpl/*')))[-1]
        # print(f'Loading npy {npy_path}')
        # npy_pose_path = os.path.join(npy_path, 'poses.npy')
        # npy_betas_path = os.path.join(npy_path, 'mean_shape.npy')
        # npy_transl_path = os.path.join(npy_path, 'normalize_trans.npy')
        # # load
        # body_pose = np.load(npy_pose_path)
        # betas = np.load(npy_betas_path)
        # transl = np.load(npy_transl_path)

        camPs = np.load(os.path.join(v2a_root, 'cameras.npz'))
        norm_camPs = np.load(os.path.join(v2a_root, 'cameras_normalize.npz'))
        scale = norm_camPs['scale_mat_0'].astype(np.float32)[0,0]
        extrinsics = []
        intrinsics = None
        image_files = sorted(glob.glob(os.path.join(v2a_root, "image/*.png")))
        print("number of image ",len(image_files))
        for i in range(len(image_files)):
            out = cv2.decomposeProjectionMatrix(camPs[f'cam_{i}'][:3, :])
            if i == 0:
                intrinsics = out[0]

            render_R = out[1]
            cam_center = out[2]
            cam_center = (cam_center[:3] / cam_center[3])[:, 0]
            render_T = -render_R @ cam_center
            ext = np.zeros((4, 4))
            ext[:3, :3] = render_R
            ext[:3, 3] = render_T
            extrinsics.append(ext[np.newaxis])
        extrinsics = np.concatenate(extrinsics, axis=0)[:, :3]
        img_temp = cv2.imread(image_files[0])
        cols, rows = img_temp.shape[1], img_temp.shape[0]
        # cameras = OpenCVCamera(intrinsics, extrinsics, cols=cols, rows=rows, viewer=v)
        # bb = Billboard.from_camera_and_distance(cameras, 10.0, cols, rows, image_files)
        # v.scene.add(cameras, bb)



        # load one frames
        # frame_id = 131
        # frame_id = 85
        frame_id = 75
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
                                poses_root=body_pose[:, 1, :3], # + np.array([0.0, -0.03, 0.0]),
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
        # camera_center_list = []
        number_person = len(sorted(glob.glob(os.path.join(f'{v2a_output_root}/test_mesh_res4/*'))))
        print("number_person ",number_person)
        for start_frame in range(0,371):
        # for start_frame in range(100,371):
            print("start_frame ",start_frame)
            extrinsics = []
            intrinsics = None
            mesh_list = []
            for p in range(number_person):
                vertices = []
                faces = []
                vertex_normals = []
                uvs = []
                texture_paths = []
                num_frame = len(sorted(glob.glob(os.path.join(f'{v2a_output_root}/test_mesh_res4/0/*')))) // 2
                print("number_frame ",num_frame)
                # for idx in range(num_frame):
                end_frame = start_frame + 1
                # start_frame = 35
                # end_frame = 124
                # start_frame = 124
                # end_frame = 244
                # start_frame = 244
                # end_frame = 373
                # start_frame = 260
                # end_frame = 300
                # num_frame = 20
                # y_offset = [0.27 for _ in range(7)] + [(1-i/7)*0.27 + (i/7)*0.16 for i in range(7)] + [0.16 for _ in range(6)] + [(1-i/7)*0.16 + (i/7)*0.02 for i in range(7)] + [(1-i/6)*0.02 - (i/6)*0.07 for i in range(6)] +  [-(1-i/8)*0.07 for i in range(8)] + [0 for _ in range(34)]
                for idx in range(start_frame, end_frame):
                # for idx in range(num_frame):
                    mesh = trimesh.load(os.path.join(f'{v2a_output_root}/test_mesh_res4/{p}/{idx:04d}_deformed.ply'), process=False)
                    if p == 0:
                        mean_list.append(mesh.vertices.mean(axis=0))
                    mesh.vertices = mesh.vertices - mean_list[idx-0]
                    scaled_vertices = mesh.vertices * scale
                    # scaled_vertices = scaled_vertices + np.array([0, y_offset[idx], 0])
                    vertices.append(scaled_vertices)
                    faces.append(mesh.faces)
                    vertex_normals.append(mesh.vertex_normals)
                    if p == 0:
                        out = cv2.decomposeProjectionMatrix(camPs[f'cam_{idx}'][:3, :])
                        # if idx == 0:
                        if idx == start_frame:
                            intrinsics = out[0]
                        render_R = out[1]
                        cam_center = out[2]
                        cam_center = (cam_center[:3] / cam_center[3])[:, 0]
                        cam_center = cam_center - (mean_list[idx-0] * scale)
                        # camera_center_list.append(cam_center)

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
                mesh_list.append(meshes)
                v.scene.add(meshes)
            
            # image_files = image_files[:num_frame]
            image_files_subset = image_files[start_frame: end_frame]
            extrinsics = np.concatenate(extrinsics, axis=0)[:, :3]
            img_temp = cv2.imread(image_files_subset[0])
            cols, rows = img_temp.shape[1], img_temp.shape[0]
            cameras = OpenCVCamera(intrinsics, extrinsics, cols=cols, rows=rows, viewer=v)
            bb = Billboard.from_camera_and_distance(cameras, 10.0, cols, rows, image_files_subset)
            # v.scene.add(cameras, bb)
            # v.scene.add(cameras)

            print("current rendering frame id: ", v.scene.current_frame_id)
            # v.set_temp_camera(cameras)

            # camera = PinholeCamera(np.array([-4.456,0.142,2.985]), np.array([0,0,0]), 1920, 1080, 45, viewer=v)
            # az_delta = np.radians(90) / num_frame
            # camera.rotate_azimuth(az_delta)

            # v.scene.add(camera)
            # v.set_temp_camera(camera)

            # print(v.scene.camera.view_matrix)
            # az_delta = np.radians(500) / num_frame # dance5
            # if start_frame % 100 < 50:
            #     pass
            # else:
            #     az_delta = -az_delta
            # az_delta = np.radians(240) / num_frame # updown 2 round
            # if start_frame % 170 < 85:
            #     pass
            # else:
            #     az_delta = -az_delta
            
            # az_delta = np.radians(100) / num_frame # updown 1 round

            # az_delta = np.radians(720) / num_frame # dance5
            # if start_frame % 100 < 50:
            #     pass
            # else:
            #     az_delta = -az_delta

            az_delta = np.radians(100) / num_frame # updown 1 round

            print(v.scene.floor.position)
            v.scene.camera.update_matrices(1920, 1080)
            v.scene.camera.rotate_azimuth(az_delta)
            print("target", v.scene.camera.target)
            os.makedirs(os.path.join("/media/ubuntu/hdd/V2A_output/alehug_MLP_feet/ait_frame_NVS"), exist_ok=True)
            v.scene.floor.position = np.array([0, -1, 0])
            v.save_frame(os.path.join("/media/ubuntu/hdd/V2A_output/alehug_MLP_feet/ait_frame_NVS", "{:06d}.png".format(start_frame)))
            # v.scene.remove(camera)
            v.scene.remove(cameras)
            v.scene.remove(bb)

            for mesh in mesh_list:
                v.scene.remove(mesh)


    # v.run()


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
    # v2a_vanilla_root = r"Y:\cheguo\EMDB_V2A\in-stage_checkpoints"
    # from sequences import eval_sequences
    # for root, subject_id, seq_id in eval_sequences:
        # print("\nProcessing sequence: {}/{}".format(subject_id, seq_id))

        # copy_v2a_checkpoint(root, subject_id, seq_id, v2a_vanilla_root)

        # capture_root = os.path.join(root, subject_id, seq_id)
        # v2a_input_root = os.path.join(capture_root, "v2a")
        # v2a_output_root = os.path.join(capture_root, "v2a_baseline")
    C.update_conf({"smplx_models": '/media/ubuntu/hdd/Motion_infilling_smoothing/SmoothNet/data/'})
    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/updown_sam_delay_depth_loop_noshare'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/updown'
    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/pair19_gt'
    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/Hi4D_pair19_piggyback19_4_sam_delay_depth_2_noshare'
    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/Hi4D_pair19_piggyback19_4_2_noshare'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/pair19_piggyback19_vitpose_4'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/Hi4D_pair19_piggyback19_static'



    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/pair15_fight15_vitpose_4'
    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_2_noshare'
    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/Hi4D_pair15_fight15_4_sam_loop_vitpose_2_noshare'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/Hi4D_pair15_fight15_static'
    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/pair15_gt'

    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/Hi4D_pair16_jump16_4_sam_loop_align_vitpose_noshare'
    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2_edge_noshare'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/pair16_jump16_vitpose_4'

    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/cvpr_sam_delay_depth_loop_MLP'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/cvpr'

    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/teaser6_sam_delay_depth_loop_noshare'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/teaser6'

    v2a_output_root = f'/media/ubuntu/hdd/V2A_output/alehug_MLP_feet'
    v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/alehug_vitpose_openpose'


    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/dance5_sam_delay_depth_loop_noshare'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/dance5'

    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/updown_sam_delay_depth_loop_noshare_vitpose_openpose_2048'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/updown_vitpose_openpose'



    # v2a_output_root = f'/media/ubuntu/hdd/V2A_output/taichi01_sam_delay_depth_loop_2_MLP_vitpose_openpose'
    # v2a_input_root = f'/media/ubuntu/hdd/RGB-PINA/data/taichi01_vitpose_openpose'
    main_without_extrinsics(v2a_input_root, v2a_output_root, visualize_smpl=False, visualize_mesh=True)
    # main_without_extrinsics(v2a_input_root, v2a_output_root, visualize_smpl=True, visualize_mesh=False)