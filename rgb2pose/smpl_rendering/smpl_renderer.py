import torch
import torch.nn as nn
import numpy as np
import sys
from .smplx.body_models import SMPL

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
from pytorch3d.renderer.mesh import TexturesUV
from pytorch3d.io import load_obj, save_obj
from pytorch3d.io.utils import _read_image
import cv2
import matplotlib.pyplot as plt

materials = Materials(ambient_color=((0.7, 0.7, 0.7),), device='cuda')
# def rectify_pose(pose):
#     """
#     Rectify "upside down" people in global coord
 
#     Args:
#         pose (72,): Pose.

#     Returns:
#         Rotated pose.
#     """
#     pose = pose.copy()
#     R_mod = cv2.Rodrigues(np.array([np.pi, 0, 0]))[0]
#     R_root = cv2.Rodrigues(pose[:3])[0]
#     new_root = R_root.dot(R_mod)
#     pose[:3] = cv2.Rodrigues(new_root)[0].reshape(3)
#     return pose

from typing import NamedTuple, Sequence

class SMPLBlendPar(NamedTuple):
    sigma: float = 1e-9 # 0 sys.float_info.epsilon
    gamma: float = 1e-4
    background_color: Sequence = (1.0, 1.0, 1.0)

class SMPLRenderer(nn.Module):
    def __init__(self, batch_size, image_size, f=50, principal_point=((0.0, 0.0),), R=None, t=None, model_path='/home/chen/pytorch3d_xu/pytorch3d/smpl/smpl_model', gender='female'):

        super().__init__()
        self.batch_size = batch_size
        self.image_size = image_size
        self.f = f
        self.principal_point = principal_point
        self.smpl_mesh = SMPL(model_path=model_path, batch_size = batch_size, gender=gender)

        self.v_template = self.smpl_mesh.v_template.cuda()
        self.blend_params = SMPLBlendPar()

        if R is None and t is None: 
            self.cam_R = torch.from_numpy(np.array([[-1., 0., 0.],
                                                    [0., -1., 0.],
                                                    [0., 0., 1.]])).cuda().float().unsqueeze(0)

            
            self.cam_T = torch.zeros((1,3)).cuda().float()

            # no adaption to PyTorch3D needed

        if R is None and t is not None:
            self.cam_R = torch.from_numpy(np.array([[-1., 0., 0.],
                                                    [0., -1., 0.],
                                                    [0., 0., 1.]])).cuda().float().unsqueeze(0)
            self.cam_T = t.detach().clone()  
            self.cam_T[:, :2] *= -1.0
        else:
            # using the 'cam_poses' from 3DPW
            self.cam_R = R.detach().clone()
            self.cam_T = t.detach().clone()     


            # coordinate system adaption to PyTorch3D
            self.cam_R[:, :2, :] *= -1.0
            self.cam_T[:, :2] *= -1.0
            self.cam_R = torch.transpose(self.cam_R,1,2)

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        torch.cuda.set_device(self.device)

        self.cameras = SfMPerspectiveCameras(focal_length=self.f, principal_point=self.principal_point, R=self.cam_R, T=self.cam_T, device=self.device)

        self.raster_settings = RasterizationSettings(image_size=image_size,faces_per_pixel=10, blur_radius=0, perspective_correct=True) # np.log(1. / 1e-4 - 1.) * self.blend_params.sigma

        self.rasterizer = MeshRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)

        self.faces = torch.from_numpy(self.smpl_mesh.faces.astype(int)).unsqueeze(0).cuda()

        self.mask_shader = SoftSilhouetteShader(blend_params=self.blend_params)
        lights = PointLights(device=self.device, location=[[0.0, 0.0, 2.0]])
        tex_lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),), device=self.device)

        self.vis_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=SoftPhongShader(
                                                                                            device=self.device, 
                                                                                            cameras=self.cameras,
                                                                                            lights=lights,
                                                                                            materials=materials
                                                                                           ))
        self.tex_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=SoftPhongShader(
                                                                                            device=self.device, 
                                                                                            cameras=self.cameras,
                                                                                            lights=tex_lights,
                                                                                            materials=materials
                                                                                           ))        
        self.mask_renderer = MeshRenderer(rasterizer=self.rasterizer, shader=self.mask_shader)

        verts_rgb = torch.ones_like(self.v_template.unsqueeze(0)).cuda()

        self.mask_texture = TexturesVertex(verts_features=verts_rgb)

        _, faces, aux = load_obj('/home/chen/Semester-Project/smpl_rendering/text_uv_coor_smpl.obj', load_textures=True)
        # import pdb
        # pdb.set_trace()
        self.verts_uvs = aux.verts_uvs.expand(batch_size, -1, -1).to(self.device)
        self.faces_uvs = faces.textures_idx.expand(batch_size, -1, -1).to(self.device)

    # get the individual T_hip for each person
    def get_T_hip(self, betas=None, displacement=None):
        return self.smpl_mesh.get_T_hip(betas, displacement)
    
    def get_smpl_output(self, betas=None, thetas=None, trans=None, displacement=None, absolute_displacement=False):

        if displacement is not None and absolute_displacement:
            displacement = displacement - self.v_template

        smpl_output = self.smpl_mesh.forward(betas=betas, body_pose=thetas[:,3:],
                                             transl=trans,
                                             global_orient=thetas[:,:3],
                                             displacement=displacement,
                                             return_verts=True)
        return smpl_output
    def side_view(self, smpl_vertices, trans, texture=None):
        elev = torch.linspace(0, 360, 20) # torch.tensor([0]*20)
        azim = torch.tensor([-120]*20)  # torch.linspace(-180, 180, 20)  

        R, T = look_at_view_transform(dist=2.5, elev=elev, azim=azim)
        R[:, :2, :] *= -1.0
        T[:, :2] *= -1.0
        R = torch.transpose(R, 1,2)

        camera = OpenGLPerspectiveCameras(device=self.device, R=R[None, 0, ...], T=T[None, 0, ...])
        # sideV_renderer = MeshRenderer(rasterizer=MeshRasterizer(cameras=self.cameras, 
        #                                                         raster_settings=self.raster_settings
        #                                                         ),
        #                                                         shader=SoftPhongShader(device=self.device,
        #                                                         cameras=self.cameras,
        #                                                         lights=sideV_lights)
        #                                                         )
        if texture is not None:
            sideV_lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0, 0, 0),), specular_color=((0, 0, 0),), device=self.device)
            mesh_texture = TexturesUV(maps=texture, 
                                      faces_uvs=self.faces_uvs,
                                      verts_uvs=self.verts_uvs)
            sideV_mesh = Meshes(smpl_vertices-trans, faces=self.faces.expand(self.batch_size,-1,-1), textures=mesh_texture).cuda()
            sideV_img = self.tex_renderer(sideV_mesh, cameras=camera, lights=sideV_lights)
        else:
            sideV_lights = PointLights(device=self.device, location=[[0.0, 0.0, -2.0]])
            mesh_texture = self.mask_texture
            sideV_mesh = Meshes(smpl_vertices-trans, faces=self.faces.expand(self.batch_size,-1,-1), textures=mesh_texture).cuda()
            sideV_img = self.vis_renderer(sideV_mesh, cameras=camera, lights=sideV_lights)
        
        return sideV_img

    def o_mesh_rendering(self, vertices, faces):
        o_mesh_texture = TexturesVertex(verts_features=torch.ones_like(vertices))
        mesh = Meshes(verts=vertices, faces=faces, textures=o_mesh_texture).cuda()
        # import pdb
        # pdb.set_trace()
        rendered_img = self.vis_renderer(mesh)
        return rendered_img, mesh

    def forward(self, smpl_vertices, mode='a', texture=None):
        '''
        :param shape: Bx10, double
        :param pose: Bx72, double
        :param trans: Bx3, double
        '''        

        if 'a' in mode:
            mask_mesh = Meshes(verts=smpl_vertices,faces=self.faces.expand(self.batch_size,-1,-1),textures=self.mask_texture).cuda()

            mask = self.mask_renderer(mask_mesh)[...,3] 
            # mask = self.phong_renderer(mask_mesh)[..., :3].max(-1).values
            
            # mask[:,torch.nonzero(mask)]=1.0
            return mask, mask_mesh
        elif 'tex' in mode:
            mesh_texture = TexturesUV(maps=texture,
                                      faces_uvs=self.faces_uvs,
                                      verts_uvs=self.verts_uvs)
            texture_mesh = Meshes(verts=smpl_vertices,faces=self.faces.expand(self.batch_size,-1,-1),textures=mesh_texture).cuda()

            texture_img = self.tex_renderer(texture_mesh)
            return texture_img, texture_mesh
        elif 'vis' in mode:
            # import pdb 
            # pdb.set_trace()
            vis_mesh = Meshes(verts=smpl_vertices,faces=self.faces.expand(self.batch_size,-1,-1),textures=self.mask_texture).cuda()
            rendered_img = self.vis_renderer(vis_mesh)

            # save_obj('test.obj', smpl_vertices.squeeze(), self.faces.squeeze())
            return rendered_img, vis_mesh
        else:
            raise NotImplementedError('%s mode not supported by SMPLRenderer.' % mode)


if __name__ == '__main__':
    import os.path
    
    import pickle

    batch_size = 1
    image_size = 1920

    mesh_file = '/home/chen/3DPW/sequenceFiles/test/downtown_arguing_00.pkl'
    with open(mesh_file,'rb') as f:
        file = pickle.load(f, encoding='latin1')
    for frame in range(0, file['cam_poses'].shape[0], 16):
        images=[]
        overlays = []
        img_path = os.path.join('/home/chen/3DPW/imageFiles/downtown_arguing_00/image_{:05d}.jpg'.format(frame))
        img = plt.imread(img_path)


        for actor in range(len(file['v_template_clothed'])):
            # R = torch.from_numpy(file['cam_poses'][frame][:3,:3]).cuda().float().unsqueeze(0)
            # t = torch.from_numpy(file['cam_poses'][frame][:3,3]).cuda().float().unsqueeze(0)
            cam_intrinsics = file['cam_intrinsics']
            half_max_length = max(cam_intrinsics[0:2,2])
            
            # cam_center = torch.from_numpy((cam_intrinsics[0:2,2]-half_max_length)/half_max_length).unsqueeze(0)
            # cam_center[:,1] *= -1

            f = torch.tensor([(cam_intrinsics[0,0]/half_max_length).astype(np.float32), (cam_intrinsics[1,1]/half_max_length).astype(np.float32)]).unsqueeze(0)                           
            # render_ori = SMPLRenderer(batch_size, image_size, f, R, t).cuda()
            render = SMPLRenderer(batch_size, image_size, f).cuda()
            betas = file['betas_clothed'][actor][:10]
            betas = torch.tensor(betas.astype(np.float32)).unsqueeze(0).cuda().expand(batch_size, -1)
            displacement = torch.from_numpy(file['v_template_clothed'][actor].astype(np.float32)).cuda().float() - render.v_template

            extrinsics = file['cam_poses'][frame]
            R_smpl = cv2.Rodrigues(file['poses'][actor][frame][:3])[0]
            T_smpl = file['trans'][actor][frame].reshape(3,1)

            T_hip = render.get_T_hip(betas, displacement).data.cpu().numpy().reshape(3,1)

            R_c = extrinsics[:3,:3]
            T_c = extrinsics[:3,3:]
            R_smpl = R_c @ R_smpl  
            trans = (R_c @ (T_hip + T_smpl) + T_c - T_hip).reshape(3)
            trans = torch.tensor(trans.astype(np.float32)).unsqueeze(0).cuda().expand(batch_size, -1)


            thetas = file['poses'][actor][frame].copy()
            thetas[:3] = cv2.Rodrigues(R_smpl)[0].T[0]
            # thetas = rectify_pose(thetas)
            thetas = torch.tensor(thetas.astype(np.float32)).unsqueeze(0).cuda().expand(batch_size, -1)
            smpl_output = render.get_smpl_output(betas=betas,thetas=thetas, trans=trans,
                                                 displacement=displacement.expand(batch_size,-1,-1),
                                                absolute_displacement=False)
            texture = torch.from_numpy(_read_image(file_name='texture_map.jpg', format='RGB') / 255. ).unsqueeze(0).cuda()
            image, _ = render.forward(smpl_output.vertices, mode='tex', texture=texture)

            # image = (255 * image.data.cpu()[0].numpy()).astype(np.uint8)                        
    


            images.append(image)
        import PIL.Image as pil_img
        img_path = os.path.join('/home/chen/3DPW/imageFiles/downtown_arguing_00/image_{:05d}.jpg'.format(frame))
        input_img = plt.imread(img_path).copy()
        # comb = (images[0][0,...,:3] + images[1][0,...,:3])
        # import pdb
        # pdb.set_trace()
        image = images[0][0,420:1500,...].cpu().numpy() * 255
        # plt.figure(figsize=(10,10))
        # plt.imshow((image[:, :, -1]).astype(np.uint8))
        # plt.show()
        valid_mask = (image[:, :, -1] > 0)[:, :, np.newaxis]

        output_img = (image[:, :, :-1] * valid_mask +  (1 - valid_mask) * input_img)
        plt.figure(figsize=(10,10))
        plt.imshow(output_img.astype(np.uint8)) 
        plt.show()
        # comb = comb[420:1500].cpu().numpy()
        # plt.figure(figsize=(10,10))
        # plt.imshow((image[:, :, -1]).astype(np.uint8))
        # img = pil_img.fromarray((output_img).astype(np.uint8))
        # img.save('test.png')
        # plt.show()


        
        img_rgb = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGBA)[:,:,:3]
        # img_rgb[np.where(comb==255)[0],np.where(comb==255)[1], :2]=0
        # cv2.imwrite('test_sil.png',img_rgb)


    


    
