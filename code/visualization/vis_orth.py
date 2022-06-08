import numpy as np
import cv2
import torch
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    look_at_view_transform,
    camera_position_from_spherical_angles
)

from pytorch3d.structures import Meshes
from pytorch3d.renderer.mesh import Textures
import glob
import trimesh
import pickle as pkl
from tqdm import trange, tqdm
class Renderer():
    
    def __init__(self, image_size=512):
        
        super().__init__()

        self.image_size = image_size

        self.device = torch.device("cuda:0")
        torch.cuda.set_device(self.device)

        R = torch.from_numpy(np.array([[-1., 0., 0.],
                                       [0., 1., 0.],
                                       [0., 0., -1.]])).cuda().float().unsqueeze(0)


        t = torch.from_numpy(np.array([[0., 0.3, 5.]])).cuda().float()

        self.cameras = FoVOrthographicCameras(R=R, T=t,device=self.device)

        self.lights = PointLights(device=self.device,location=[[0.0, 0.0, 3.0]],
                            ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))

        self.raster_settings = RasterizationSettings(image_size=image_size,faces_per_pixel=100,blur_radius=0, max_faces_per_bin=20000)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)
    
    def set_render(self, R, t, image_size=512):
        

        self.cameras = FoVOrthographicCameras(R=R, T=t,device=self.device)

        self.lights = PointLights(device=self.device,location=[[0.0, 0.0, 3.0]],
                            ambient_color=((1,1,1),),diffuse_color=((0,0,0),),specular_color=((0,0,0),))

        # self.raster_settings = RasterizationSettings(image_size=image_size,faces_per_pixel=100, blur_radius=0, max_faces_per_bin=200000)
        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.shader = HardPhongShader(device=self.device, cameras=self.cameras, lights=self.lights)

        self.renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.shader)
        
    def render_mesh(self, verts, faces, colors=None, mode='npat', light = None):
        '''
        mode: normal, phong, texture
        '''
        with torch.no_grad():

            mesh = Meshes(verts, faces)

            normals = torch.stack(mesh.verts_normals_list())
            front_light = light
            shades = (normals * front_light.view(1,1,3)).sum(-1).clamp(min=0).unsqueeze(-1).expand(-1,-1,3)
            results = []

            # normal
            if 'n' in mode:
                normals_vis = normals* 0.5 + 0.5 
                mesh_normal = Meshes(verts, faces, textures=Textures(verts_rgb=normals_vis))
                image_normal = self.renderer(mesh_normal)
                results.append(image_normal)

            # shading
            if 'p' in mode:
                mesh_shading = Meshes(verts, faces, textures=Textures(verts_rgb=shades))
                image_phong = self.renderer(mesh_shading)
                results.append(image_phong)

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

image_size = 512
torch.cuda.set_device(torch.device("cuda:0"))
device = torch.device("cuda:0")
renderer = Renderer(image_size)
def render_trimesh(mesh,R,T, light, mode='np'):

    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    # verts[:,:,-1] *= -1
    # verts[:,:,-1] += 1
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255
    renderer.set_render(R,T)
    image = renderer.render_mesh(verts, faces, colors=colors, mode=mode, light = light)[0]
    # import pdb
    # pdb.set_trace()
    image = (255*image).data.cpu().numpy().astype(np.uint8)
    
    return image


if __name__ == '__main__':
    
    i = 0
    step = 5
    frame = 219
    for i in trange(360//step):
        # i=0
        R, T = look_at_view_transform(2, 0, i*step%360, device=device)
        # R[:, :2, :] *= -1.0
        # T[:, :2] *= -1.0
        # R = torch.transpose(R, 1, 2)
        light = camera_position_from_spherical_angles(2,0,i*step%360, device = device)/2
        T[:,1] = T[:,1] + 0.3
        # i = i + 1

        mesh = trimesh.load('/home/chen/RGB-PINA/code/outputs/BuffMonoSeg/debug_buffmono_all/rendering/1069.ply', process=False)
        # trans = pkl.load(open('/home/chen/disk2/kinect_capture_results/%s/%s/smoothed_files/refined_smpl_params_%04d.pkl' % (actor, seq, frame), 'rb'))['trans']
        # mesh.vertices -= trans
        # rot_mat = trimesh.transformations.rotation_matrix(angle=np.radians(170), direction=[1,0,0], point=mesh.vertices.mean(0))
        # mesh.apply_transform(rot_mat)
        rendered_image = render_trimesh(mesh, R, T, light, 'p')
        # rendered_image_n = rendered_image[420:1500]
        cv2.imwrite('/home/chen/RGB-PINA/vis_results/BuffMonoSeg/buffmono_all/%04d.png' % i, rendered_image)


