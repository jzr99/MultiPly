
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
overlay = False
if __name__ == '__main__':
    device = torch.device("cuda:0")
    seq = 'courtyard_jumpBench_01'
    if overlay:
        output_dir = f'/home/chen/disk2/3DPW/vis_results/{seq}'
    else:
        output_dir = f'/home/chen/RGB-PINA/data/{seq}/mask'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_dir = f'/home/chen/disk2/3DPW/imageFiles/{seq}'   
    seq_dir = f'/home/chen/disk2/3DPW/sequenceFiles/validation/{seq}.pkl'
    img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
    seq_file = pkl.load(open(seq_dir, 'rb'), encoding='latin1')
    gender = seq_file['genders'][0]

    if gender == 'f':
        gender = 'female'
    elif gender == 'm':
        gender = 'male'
    import sys
    sys.path.append('/home/chen/RGB-PINA/rgb2pose')
    from smplx import SMPL
    smpl_model = SMPL('/home/chen/Models/smpl', gender=gender)
    smpl_shape = seq_file['betas'][0][:10]
    cam_intrinsics = seq_file['cam_intrinsics']

    input_img = cv2.imread(img_paths[0])
    renderer = Renderer(img_size = [input_img.shape[0], input_img.shape[1]], cam_intrinsic=cam_intrinsics)

    for idx, img_path in enumerate(tqdm(img_paths)):
        input_img = cv2.imread(img_path)

        smpl_pose = seq_file['poses'][0][idx]
        smpl_trans = seq_file['trans'][0][idx]
        cam_extrinsics = seq_file['cam_poses'][idx]

        R = torch.tensor(cam_extrinsics[:3,:3])[None].float()
        T = torch.tensor(cam_extrinsics[:3, 3])[None].float() 
        smpl_output = smpl_model(betas=torch.tensor(smpl_shape)[None].float(),
                                 body_pose=torch.tensor(smpl_pose[3:])[None].float(),
                                 global_orient=torch.tensor(smpl_pose[:3])[None].float(),
                                 transl=torch.tensor(smpl_trans)[None].float())
        smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

        # verts = torch.tensor(smpl_verts).cuda().float()[None]
        # faces = torch.tensor(smpl_model.faces).cuda()[None]
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
            cv2.imwrite(os.path.join(output_dir, '%04d.png' % idx), valid_mask*255)
        # import ipdb
        # ipdb.set_trace()