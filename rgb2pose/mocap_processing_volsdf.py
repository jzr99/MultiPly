import cv2
import numpy as np
import pickle as pkl
import os
import glob
import torch
from pytorch3d.transforms import matrix_to_axis_angle
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (PerspectiveCameras, RasterizationSettings,
                                MeshRasterizer, PointLights, SoftPhongShader,
                                MeshRenderer)
from pytorch3d.renderer.mesh import Textures
from pytorch3d.renderer.blending import BlendParams
import trimesh
# from normal_renderer import Renderer

def real_mocap():

    target_cam_ids = [4, 28, 52, 77]

    smpl_file_paths = sorted(glob.glob(os.path.join('/home/chen/disk2/motion_capture/otmar_waving/smpl', '*.pkl')))
    smpl_mesh_paths = sorted(glob.glob(os.path.join('/home/chen/disk2/motion_capture/otmar_waving/smpl', '*.ply')))
    smpl_poses = []
    smpl_trans = []
    smpl_betas = []
    normalize_shift_list = []

    cameras = dict(np.load(os.path.join('/home/chen/disk2/motion_capture/otmar_waving/cameras', 'rgb_cameras.npz'))) 

    for i, smpl_file_path in enumerate(smpl_file_paths):

        smpl_file = pkl.load(open(smpl_file_path, 'rb'))
        smpl_verts = trimesh.load(smpl_mesh_paths[i]).vertices
        v_max = smpl_verts.max(axis=0)
        v_min = smpl_verts.min(axis=0)
        # pose = matrix_to_axis_angle(smpl_file['full_pose'][0].detach()).reshape(-1).cpu().numpy()

        pose = smpl_file['pose'] # np.vstack([smpl_file['global_orient'][0].detach().cpu().numpy().reshape(-1, 1), smpl_file['body_pose'][0].detach().cpu().numpy().reshape(-1, 1)])[:, 0]
        transl = smpl_file['trans']
        beta = smpl_file['betas']

        smpl_poses.append(pose)
        smpl_trans.append(transl)
        smpl_betas.append(beta)
        normalize_shift_list.append(-(v_max + v_min) / 2.)


    smpl_poses = np.array(smpl_poses)
    smpl_trans = np.array(smpl_trans)
    smpl_betas = np.array(smpl_betas)
    normalize_shift_array = np.array(normalize_shift_list)

    normalize_shift = normalize_shift_array.mean(0)

    smpl_trans += normalize_shift

    mean_shape = smpl_betas.mean(0)

    cam_dict = {}

    for i in range(len(cameras['ids'])):
        cam_id = cameras['ids'][i]
        if cam_id in target_cam_ids:
            intrinsic = cameras['intrinsics'][i]
            extrinsic = cameras['extrinsics'][i]
            extrinsic[:3, -1] = extrinsic[:3, -1] - (extrinsic[:3, :3] @ normalize_shift)

            P = intrinsic @ extrinsic
            cam_dict['cam_%d' % cam_id] = P

    np.save('/home/chen/RGB-PINA/data/mocap_otmar_waiving/poses.npy', smpl_poses)
    np.save('/home/chen/RGB-PINA/data/mocap_otmar_waiving/normalize_trans.npy', smpl_trans)
    np.save('/home/chen/RGB-PINA/data/mocap_otmar_waiving/mean_shape.npy', mean_shape)

    np.savez('/home/chen/RGB-PINA/data/mocap_otmar_waiving/cameras.npz', **cam_dict)
    np.save('/home/chen/RGB-PINA/data/mocap_otmar_waiving/normalize_shift.npy', normalize_shift)

def real_mocap_normal():
    device = 'cuda:0'
    image_size=(1280, 940)
    target_cam_ids = [4, 28, 52, 77]

    scan_file_paths = sorted(glob.glob(os.path.join('/home/chen/disk2/motion_capture/otmar_waving/meshes_vis', '*.ply')))
    camera_infos = dict(np.load(os.path.join('/home/chen/disk2/motion_capture/otmar_waving/cameras', 'rgb_cameras.npz'))) 
    cameras = {}
    normal_output_dir = '/home/chen/disk2/motion_capture/otmar_waving/scan_rendered_normal'

    for i in range(len(camera_infos['ids'])):
        cam_id = camera_infos["ids"][i]
        if cam_id not in target_cam_ids: # TODO
            continue
        intrinsic = camera_infos["intrinsics"][i]
        extrinsic = camera_infos["extrinsics"][i]
        focal_length = torch.tensor([(intrinsic[0, 0]).astype(np.float32),
                                     (intrinsic[1, 1]).astype(np.float32)
                                     ]).to(device).unsqueeze(0)
        principal_point = [intrinsic[0, 2], intrinsic[1, 2]]
        principal_point = torch.tensor(principal_point).float().to(
            device).unsqueeze(0)
        cam_R = torch.from_numpy(extrinsic[:, :3]).float().to(device).unsqueeze(0)
        cam_T = torch.from_numpy(extrinsic[:,3]).float().to(device).unsqueeze(0)
        # coordinate system adaption to PyTorch3D
        cam_R[:, :2, :] *= -1.0
        cam_T[:, :2] *= -1.0  # camera position in world space -> world position in camera space
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

    cameras = dict(sorted(cameras.items()))
    for cam_id in cameras.keys():
        os.makedirs(os.path.join(normal_output_dir, cam_id), exist_ok=True)

    lights = PointLights(device=device,location=[[0.0, 0.0, 0.0]], ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))
    raster_settings = RasterizationSettings(image_size=image_size, faces_per_pixel=10, blur_radius=0, max_faces_per_bin=40000)
    
    for scan_file_path in scan_file_paths:
        scan_fn = os.path.split(scan_file_path)[1][6:11]
        scan = trimesh.load(scan_file_path, process=False)
        v = torch.from_numpy(scan.vertices).float().to(device)
        f = torch.from_numpy(scan.faces).float().to(device)
        scan_mesh = Meshes([v], [f]).to(device)
        normals = torch.stack(scan_mesh.verts_normals_list())
        normals_vis = normals * 0.5 + 0.5
        # import ipdb
        # ipdb.set_trace()
        normals_vis = normals_vis[:, :, [2,1,0]]
        mesh_normal = Meshes([v], [f], textures=Textures(verts_rgb=normals_vis))
        for cam_id, camera in cameras.items():
            rasterizer = MeshRasterizer(cameras=camera, 
                                         raster_settings=raster_settings)
            shader = SoftPhongShader(device=device, cameras=camera,
                                     lights=lights, blend_params=BlendParams(sigma=1e-4, gamma=1e-4, background_color=np.array([127, 127, 127])/255))
            renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
            image_normal = renderer(mesh_normal)[0, ..., :3].detach().cpu().numpy()

            cv2.imwrite(
                os.path.join(normal_output_dir, str(cam_id),
                         '{}.png'.format(scan_fn.zfill(6))), image_normal * 255)


def rendered_mocap():
    from smplx import SMPL
    import json
    smpl_faces = np.load('/home/chen/IPNet/faces.npy')
    gender = 'male'
    smpl_model = SMPL('/home/chen/Models/smpl', gender=gender)
    target_cam_ids = [0, 2, 4, 7]

    normalize_shift_list = []
    image_dir = '/home/chen/disk2/motion_capture/Ernst/images/0'
    image_0_paths = sorted(os.listdir(image_dir))
    smpl_params = np.load('/home/chen/disk2/motion_capture/Ernst/smpl_params.npz')
    smpl_poses = smpl_params['pose'][0]
    smpl_trans = smpl_params['trans'][0]
    smpl_betas = smpl_params['betas'][0]
    mean_shape = smpl_betas.mean(0)

    for i in range(smpl_poses.shape[0]):
        frame = int(image_0_paths[i].split('.')[0])
        smpl_output = smpl_model(betas = torch.tensor(mean_shape)[None].float(),
                                 body_pose = torch.tensor(smpl_poses[i][3:])[None].float(),
                                 global_orient = torch.tensor(smpl_poses[i][:3])[None].float(),
                                 transl = torch.tensor(smpl_trans[i])[None].float())    
        
        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
        smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_faces, process=False)
        _ = smpl_mesh.export('/home/chen/disk2/motion_capture/Ernst/smpl/mesh-f%05d_smpl.obj' % frame)
        v_max = smpl_verts.max(axis=0)
        v_min = smpl_verts.min(axis=0)

        tmp_normalize_shift = -(v_max + v_min) / 2.

        normalize_shift_list.append(tmp_normalize_shift)
    
    normalize_shift_array = np.array(normalize_shift_list)

    normalize_shift = normalize_shift_array.mean(0)

    smpl_trans += normalize_shift

    cameras = json.load(open('/home/chen/disk2/motion_capture/Ernst/calibration.json', 'r'))

    cam_dict = {}
    for idx, cam_id in enumerate(target_cam_ids):

        cam = cameras['%d' % cam_id]
        intrinsic = np.array(cam['K'])
        extrinsic = np.array(cam['RT'])
        extrinsic[:3, -1] = extrinsic[:3, -1] - (extrinsic[:3, :3] @ normalize_shift)
        P = intrinsic @ extrinsic
        cam_dict['cam_%d' % idx] = P
    np.save('/home/chen/RGB-PINA/data/mocap_ernst/poses.npy', smpl_poses)
    np.save('/home/chen/RGB-PINA/data/mocap_ernst/normalize_trans.npy', smpl_trans)
    np.save('/home/chen/RGB-PINA/data/mocap_ernst/mean_shape.npy', mean_shape)

    np.savez('/home/chen/RGB-PINA/data/mocap_ernst/cameras.npz', **cam_dict)
    np.save('/home/chen/RGB-PINA/data/mocap_ernst/normalize_shift.npy', normalize_shift)
if __name__ == '__main__':
    real_mocap()