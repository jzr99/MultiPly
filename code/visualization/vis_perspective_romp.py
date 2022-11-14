
from turtle import width
from matplotlib.pyplot import contour
from pytorch3d.renderer import (
    SfMPerspectiveCameras,
    OpenGLPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    SoftSilhouetteShader,
    SoftPhongShader,
    HardPhongShader,
    PointLights,
    TexturesVertex,
    Materials,
    look_at_view_transform
)
from sklearn.neighbors import NearestNeighbors
from pytorch3d.structures import Meshes
import numpy as np
import torch
import trimesh
from pytorch3d.renderer.mesh import Textures
import cv2
import os
from tqdm import tqdm
import glob
import pickle as pkl
class Renderer():
    
    def __init__(self, focal_length=None, principal_point=None, img_size=None, cam_intrinsic = None):
    
        super().__init__()
        # img_size=[1080, 1920]
        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)
        self.cam_intrinsic = cam_intrinsic
        self.image_size = img_size
        self.render_img_size = np.max(img_size)

        principal_point = [-(self.cam_intrinsic[0,2]-self.image_size[1]/2.)/(self.image_size[1]/2.), -(self.cam_intrinsic[1,2]-self.image_size[0]/2.)/(self.image_size[0]/2.)]  
        self.principal_point = torch.tensor(principal_point, device=self.device).unsqueeze(0)

        self.cam_R = torch.from_numpy(np.array([[-1., 0., 0.],
                                                [0., -1., 0.],
                                                [0., 0., 1.]])).cuda().float().unsqueeze(0)


        self.cam_T = torch.zeros((1,3)).cuda().float()

        half_max_length = max(self.cam_intrinsic[0:2,2])
        self.focal_length = torch.tensor([(self.cam_intrinsic[0,0]/half_max_length).astype(np.float32), \
                        (self.cam_intrinsic[1,1]/half_max_length).astype(np.float32)]).unsqueeze(0)
        
        self.cameras = SfMPerspectiveCameras(focal_length=self.focal_length, principal_point=self.principal_point, R=self.cam_R, T=self.cam_T, device=self.device)

        self.lights = PointLights(device=self.device,location=[[0.0, 0.0, 0.0]], ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))
        # self.lights = PointLights(device=self.device, location=[[0.0, 0.0, 2.0]])
        self.raster_settings = RasterizationSettings(image_size=self.render_img_size, faces_per_pixel=10, blur_radius=0, max_faces_per_bin=30000)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)
    
    def set_camera(self, R, T):
        self.cam_R = R
        self.cam_T = T
        self.cam_R[:, :2, :] *= -1.0
        self.cam_T[:, :2] *= -1.0
        self.cam_R = torch.transpose(self.cam_R,1,2)
        self.cameras = SfMPerspectiveCameras(focal_length=self.focal_length, principal_point=self.principal_point, R=self.cam_R, T=self.cam_T, device=self.device)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.shader = SoftPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)
        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)

    def render_mesh_recon(self, verts, faces, R=None, T=None, colors=None, mode='npat'):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = -torch.tensor([0,0,-1]).float().to(verts.device)
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []
            # shading
            if 'p' in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)
            # normal
            if 'n' in mode:
                # import pdb
                # pdb.set_trace()

                normals_vis = normals* 0.5 + 0.5 # -1*normals* 0.5 + 0.5 
                normals_vis = normals_vis[:,:,[2,1,0]]
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)



            # albedo
            if 'a' in mode: 
                assert(colors is not None)
                mesh_albido = Meshes(verts, faces, textures=Textures(verts_rgb=colors))
                image_color = self.renderer(mesh_albido)
                results.append(image_color)
            
            # albedo*shading
            if 't' in mode: 
                assert(colors is not None)
                mesh_teture = Meshes(verts, faces, textures=Textures(verts_rgb=colors*shades))
                image_color = self.renderer(mesh_teture)
                results.append(image_color)

            return  torch.cat(results, axis=1)

def render_trimesh(mesh,R,T, mode='np'):
    
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255
    renderer.set_camera(R,T)
    image = renderer.render_mesh_recon(verts, faces, colors=colors, mode=mode)[0]
    image = (255*image).data.cpu().numpy().astype(np.uint8)
    
    return image

def get_camera_parameters(pred_cam, bbox, cam_trans):
    FOCAL_LENGTH = 5000.
    CROP_SIZE = 224

    bbox_cx, bbox_cy, bbox_w, bbox_h = bbox
    assert bbox_w == bbox_h

    bbox_size = bbox_w
    bbox_x = bbox_cx - bbox_w / 2.
    bbox_y = bbox_cy - bbox_h / 2.

    scale = bbox_size / CROP_SIZE

    cam_intrinsics = np.eye(3)
    cam_intrinsics[0, 0] = FOCAL_LENGTH * scale
    cam_intrinsics[1, 1] = FOCAL_LENGTH * scale
    cam_intrinsics[0, 2] = bbox_size / 2. + bbox_x 
    cam_intrinsics[1, 2] = bbox_size / 2. + bbox_y

    cam_s, cam_tx, cam_ty = pred_cam
    trans = [cam_tx, cam_ty, 2*FOCAL_LENGTH/(CROP_SIZE*cam_s + 1e-9)]

    cam_extrinsics = np.eye(4)
    cam_extrinsics[:3, 3] = trans

    return cam_intrinsics, cam_extrinsics

def estimate_translation_cv2(joints_3d, joints_2d, focal_length=600, img_size=np.array([512.,512.]), proj_mat=None, cam_dist=None):
    if proj_mat is None:
        camK = np.eye(3)
        camK[0,0], camK[1,1] = focal_length, focal_length
        camK[:2,2] = img_size//2
    else:
        camK = proj_mat
    ret, rvec, tvec,inliers = cv2.solvePnPRansac(joints_3d, joints_2d, camK, cam_dist,\
                              flags=cv2.SOLVEPNP_EPNP,reprojectionError=20,iterationsCount=100)

    if inliers is None:
        return INVALID_TRANS
    else:
        tra_pred = tvec[:,0]            
        return tra_pred

overlay = False
if __name__ == '__main__':
    device = torch.device("cuda:0")
    seq = 'beautiful_disaster'
    dataset = 'youtube' # 'youtube' 'monoperfcap' # 'neuman # threedpw
    transpose = False
    gender = 'm'
    if dataset == 'youtube' or dataset == 'neuman' or dataset == 'threedpw' or dataset == 'synthetic':
        DIR = '/home/chen/disk2/Youtube_Videos'
    elif dataset == 'monoperfcap':
        DIR = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset'
    elif dataset == 'deepcap':
        DIR = '/home/chen/disk2/MPI_INF_Dataset/DeepCapDataset'
    output_dir = f'{DIR}/{seq}/init_mask'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_dir = f'{DIR}/{seq}/frames'   
    file_dir = f'{DIR}/{seq}/ROMP'
    img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    file_paths = sorted(glob.glob(f"{file_dir}/*.npz"))
    

    if gender == 'f':
        gender = 'female'
    elif gender == 'm':
        gender = 'male'
    import sys
    sys.path.append('/home/chen/RGB-PINA/rgb2pose')
    from smplx import SMPL
    smpl_model = SMPL('/home/chen/Models/smpl', gender=gender)
    
    input_img = cv2.imread(img_paths[0])

    if dataset == 'youtube':
        focal_length = 1920 # 640 # 1920 # 1280 # 995.55555556
        if transpose:
            cam_intrinsics = np.array([[focal_length, 0., 540.],[0.,focal_length, 960.],[0.,0.,1.]])
        else:
            cam_intrinsics = np.array([[focal_length, 0., 960.],[0.,focal_length, 540.],[0.,0.,1.]]) # np.array([[focal_length, 0., 320.],[0.,focal_length, 180.],[0.,0.,1.]]) # np.array([[focal_length, 0., 960.],[0.,focal_length, 540.],[0.,0.,1.]]) # np.array([[focal_length, 0., 640.],[0.,focal_length, 360.],[0.,0.,1.]])
    elif dataset == 'synthetic':
        focal_length = 1920
        cam_intrinsics = np.array([[focal_length, 0., 960.],[0.,focal_length, 510.],[0.,0.,1.]])

    elif dataset == 'neuman':
        with open(f'/home/chen/disk2/NeuMan_dataset/{seq}/sparse/cameras.txt') as f:
            lines = f.readlines()
        cam_params = lines[3].split()
        cam_intrinsics = np.array([[float(cam_params[4]), 0., float(cam_params[6])], 
                                   [0., float(cam_params[5]), float(cam_params[7])], 
                                   [0., 0., 1.]])
    elif dataset == 'monoperfcap':
        # focal_length = None
        # with open(f'/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/{seq}/calib.txt') as f:
        #     lines = f.readlines()
        # cam_params = lines[2].split()
        # cam_intrinsics = np.array([[float(cam_params[1]), 0., float(cam_params[3])], 
        #                            [0., float(cam_params[6]), float(cam_params[7])], 
        #                            [0., 0., 1.]])
        focal_length = 1920 # 1280 # 995.55555556
        cam_intrinsics = np.array([[focal_length, 0., 960.],[0.,focal_length, 540.],[0.,0.,1.]])
    elif dataset == 'deepcap':
        with open(f'/home/chen/disk2/MPI_INF_Dataset/DeepCapDataset/monocularCalibrationBM.calibration') as f:
            lines = f.readlines()

        cam_params = lines[5].split()
        cam_intrinsics = np.array([[float(cam_params[1]), 0., float(cam_params[3])], 
                                   [0., float(cam_params[6]), float(cam_params[7])], 
                                   [0., 0., 1.]])
    # elif dataset == 'threedpw':
    #     source_dir = f'/home/chen/disk2/3DPW/sequenceFiles/test/outdoors_fencing_01.pkl'
    #     source_file = pkl.load(open(source_dir, 'rb'), encoding='latin1')
    #     cam_intrinsics = source_file['cam_intrinsics']
    renderer = Renderer(img_size = [input_img.shape[0], input_img.shape[1]], cam_intrinsic=cam_intrinsics)
    last_j3d = None
    for idx, img_path in enumerate(tqdm(img_paths)):
        input_img = cv2.imread(img_path)
        seq_file = np.load(file_paths[idx], allow_pickle=True)['results'][()]
        actor_id = 0

        # tracking in case of two persons
        if len(seq_file['smpl_thetas']) >= 2:
            dist = []
            for i in range(len(seq_file['smpl_thetas'])):
                dist.append(np.linalg.norm(seq_file['joints'][i].mean(0) - last_j3d.mean(0, keepdims=True)))
            # dist = [np.linalg.norm(seq_file['pj2d_org'][0].mean(0) - last_pj2d.mean(0, keepdims=True)), np.linalg.norm(seq_file['pj2d_org'][1].mean(0) - last_pj2d.mean(0, keepdims=True))]
            actor_id = np.argmin(dist)
        smpl_pose = seq_file['smpl_thetas'][actor_id]
        smpl_trans = [0.,0.,0.] # seq_file['trans'][0][idx]
        smpl_shape = seq_file['smpl_betas'][actor_id][:10]
        cam = seq_file['cam'][actor_id]
        smpl_verts = seq_file['verts'][actor_id]
        # cam_trans = seq_file['cam_trans'][0]
        pj2d_org = seq_file['pj2d_org'][actor_id]
        joints3d = seq_file['joints'][actor_id]
        last_j3d = joints3d.copy()
        tra_pred = estimate_translation_cv2(joints3d, pj2d_org, proj_mat=cam_intrinsics)

        cam_extrinsics = np.eye(4)
        cam_extrinsics[:3, 3] = tra_pred # cam_trans

        # cam_extrinsics[:3, 3] = cam_extrinsics[:3, 3] - (cam_extrinsics[:3, :3] @ normalize_shift)
        # P = cam_intrinsics @ cam_extrinsics[:3, :]
        

        R = torch.tensor(cam_extrinsics[:3,:3])[None].float()
        T = torch.tensor(cam_extrinsics[:3, 3])[None].float() 
        # smpl_output = smpl_model(betas=torch.tensor(smpl_shape)[None].float(),
        #                          body_pose=torch.tensor(smpl_pose[3:])[None].float(),
        #                          global_orient=torch.tensor(smpl_pose[:3])[None].float(),
        #                          transl=torch.tensor(smpl_trans)[None].float())
        # smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
        

        smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
        rendered_image = render_trimesh(smpl_mesh, R, T, 'n')
        if input_img.shape[0] < input_img.shape[1]:
            rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
        else:
            rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]   
        valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
        if overlay:
            output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
            cv2.imwrite(os.path.join(output_dir, '%04d.png' % idx), output_img)
        else:
            # output_dir = f'/home/chen/RGB-PINA/data/{seq}/mask_ori'
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
            # cv2.imwrite(os.path.join(output_dir, '%04d.png' % idx), output_img)
            cv2.imwrite(os.path.join(output_dir, '%04d.png' % idx), valid_mask*255)
        # import ipdb
        # ipdb.set_trace()