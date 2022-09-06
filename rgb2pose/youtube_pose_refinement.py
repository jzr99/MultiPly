
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
from torch import unsqueeze
import trimesh
from pytorch3d.renderer.mesh import Textures
import cv2
import os
from tqdm import tqdm
import glob
import pickle as pkl
from tqdm import tqdm
from utils import smpl_to_pose, PerspectiveCamera
from loss import joints_2d_loss, pose_temporal_loss, pose_prior_loss, foot_prior_loss, get_loss_weights
smpl2op_mapping = torch.tensor(smpl_to_pose(model_type='smpl', use_hands=False, use_face=False,
                                            use_face_contour=False, openpose_format='coco25'), dtype=torch.long).cuda()

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
        cam_R = R.clone()
        cam_T = T.clone()
        cam_R[:, :2, :] *= -1.0
        cam_T[:, :2] *= -1.0
        cam_R = torch.transpose(cam_R,1,2)
        self.cameras = SfMPerspectiveCameras(focal_length=self.focal_length, principal_point=self.principal_point, R=cam_R, T=cam_T, device=self.device)
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


if __name__ == '__main__':
    device = torch.device("cuda:0")
    seq = 'parkinglot'
    dataset = 'neuman' # 'youtube' 'monoperfcap' 'neuman'
    gender = 'm'
    if dataset == 'youtube' or dataset == 'neuman':
        DIR = '/home/chen/disk2/Youtube_Videos'
    elif dataset == 'monoperfcap':
        DIR = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset'
    openpose_dir = f'{DIR}/{seq}/openpose'
    if not os.path.exists(f'{DIR}/{seq}/init_refined_smpl'):
        os.makedirs(f'{DIR}/{seq}/init_refined_smpl')
    if not os.path.exists(f'{DIR}/{seq}/init_refined_mask'):
        os.makedirs(f'{DIR}/{seq}/init_refined_mask')
    if not os.path.exists(f'{DIR}/{seq}/init_refined_smpl_files'):
        os.makedirs(f'{DIR}/{seq}/init_refined_smpl_files')
    img_dir = f'{DIR}/{seq}/frames'   
    file_dir = f'{DIR}/{seq}/ROMP'
    img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    file_paths = sorted(glob.glob(f"{file_dir}/*.npz"))
    openpose_paths = sorted(glob.glob(f"{openpose_dir}/*.npy"))

    if gender == 'f':
        gender = 'female'
    elif gender == 'm':
        gender = 'male'
    import sys
    sys.path.append('/home/chen/RGB-PINA/rgb2pose')
    from smplx import SMPL
    smpl_model = SMPL('/home/chen/Models/smpl', gender=gender).to(device)
    
    input_img = cv2.imread(img_paths[0])

    if dataset == 'youtube':
        focal_length = 1920 # 1280 # 995.55555556
        cam_intrinsics = np.array([[focal_length, 0., 960.],[0.,focal_length, 540.],[0.,0.,1.]])
    elif dataset == 'neuman':
        with open(f'/home/chen/disk2/NeuMan_dataset/{seq}/sparse/cameras.txt') as f:
            lines = f.readlines()
        cam_params = lines[3].split()
        cam_intrinsics = np.array([[float(cam_params[4]), 0., float(cam_params[6])], 
                                   [0., float(cam_params[5]), float(cam_params[7])], 
                                   [0., 0., 1.]])
    elif dataset == 'monoperfcap':
        # with open(f'/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/{seq}/calib.txt') as f:
        #     lines = f.readlines()
        # cam_params = lines[2].split()
        # cam_intrinsics = np.array([[float(cam_params[1]), 0., float(cam_params[3])], 
        #                            [0., float(cam_params[6]), float(cam_params[7])], 
        #                            [0., 0., 1.]])
        focal_length = 1920 # 1280 # 995.55555556
        cam_intrinsics = np.array([[focal_length, 0., 960.],[0.,focal_length, 540.],[0.,0.,1.]])
    cam_extrinsics = np.eye(4)
    render_R = torch.tensor(cam_extrinsics[:3,:3])[None].float()
    render_T = torch.tensor(cam_extrinsics[:3, 3])[None].float() 
  
    renderer = Renderer(img_size = [input_img.shape[0], input_img.shape[1]], cam_intrinsic=cam_intrinsics)
    cam = PerspectiveCamera(focal_length_x=torch.tensor(cam_intrinsics[0, 0], dtype=torch.float32),
                            focal_length_y=torch.tensor(cam_intrinsics[1, 1], dtype=torch.float32),
                            center=torch.tensor(cam_intrinsics[0:2, 2]).unsqueeze(0)).to(device)
    weight_dict = get_loss_weights()
    overlay = True
    smooth = False
    skip_optim = False
    mean_shape = []
    if not skip_optim:
        for idx, img_path in enumerate(tqdm(img_paths)):
            input_img = cv2.imread(img_path)
            seq_file = np.load(file_paths[idx], allow_pickle=True)['results'][()]
            openpose = np.load(openpose_paths[idx])
            openpose[:, -1][openpose[:, -1] < 0.01] = 0.

            smpl_pose = seq_file['smpl_thetas'][0]
            # smpl_trans = [0.,0.,0.] # seq_file['trans'][0][idx]
            smpl_shape = seq_file['smpl_betas'][0][:10]
            smpl_verts = seq_file['verts'][0]
            pj2d_org = seq_file['pj2d_org'][0]
            joints3d = seq_file['joints'][0]
            # tranform to perspective projection
            tra_pred = estimate_translation_cv2(joints3d, pj2d_org, proj_mat=cam_intrinsics)

            # cam_extrinsics[:3, 3] = tra_pred # cam_trans
            smpl_trans = tra_pred
            P = cam_intrinsics @ cam_extrinsics[:3, :]
            
            num_iters=150

            openpose_j2d = torch.tensor(openpose[:, :2][None], dtype=torch.float32, requires_grad=False, device=device)
            openpose_conf = torch.tensor(openpose[:, -1][None], dtype=torch.float32, requires_grad=False, device=device)

            opt_betas = torch.tensor(smpl_shape[None], dtype=torch.float32, requires_grad=True, device=device)
            opt_pose = torch.tensor(smpl_pose[None], dtype=torch.float32, requires_grad=True, device=device)
            opt_trans = torch.tensor(smpl_trans[None], dtype=torch.float32, requires_grad=True, device=device)

            opt_params = [{'params': opt_betas, 'lr': 1e-3},
                          {'params': opt_pose, 'lr': 1e-3},
                          {'params': opt_trans, 'lr': 1e-3}]
            optimizer = torch.optim.Adam(opt_params, lr=2e-3, betas=(0.9, 0.999))
            if idx == 0:
                last_pose = [opt_pose.detach().clone()]
            loop = tqdm(range(num_iters))
            for it in loop:
                tmp_img = input_img.copy()
                optimizer.zero_grad()

                smpl_output = smpl_model(betas=opt_betas,
                                        body_pose=opt_pose[:,3:],
                                        global_orient=opt_pose[:,:3],
                                        transl=opt_trans)
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()

                smpl_joints_2d = cam(torch.index_select(smpl_output.joints, 1, smpl2op_mapping))
                
                # for jth in range(0, smpl_joints_2d.shape[1]):
                #     output_img = cv2.circle(tmp_img, tuple(smpl_joints_2d[0].data.cpu().numpy().astype(np.int32)[jth, :2]), 3, (0,0,255), -1)
                # cv2.imwrite('{DIR}/{seq}/init_refined_smpl/smpl_2d_%04d.png' % it, output_img)

                loss = dict()
                loss['J2D_Loss'] = joints_2d_loss(openpose_j2d, smpl_joints_2d, openpose_conf)
                loss['Temporal_Loss'] = pose_temporal_loss(last_pose[0], opt_pose)
                # loss['FOOT_Prior_Loss'] = foot_prior_loss(opt_pose[:, 21:27])
                # loss['Prior_Loss'] = pose_prior_loss(opt_pose[:, 3:], opt_betas)
                w_loss = dict()
                for k in loss:
                    w_loss[k] = weight_dict[k](loss[k], it)
                
                tot_loss = list(w_loss.values())
                tot_loss = torch.stack(tot_loss).sum()
                tot_loss.backward()
                optimizer.step()

                l_str = 'Iter: %d' % it
                for k in loss:
                    l_str += ', %s: %0.4f' % (k, weight_dict[k](loss[k], it).mean().item())
                    loop.set_description(l_str)     
                # smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)

                # smpl_mesh.export('{DIR}/{seq}/init_refined_smpl/%04d.ply' % it)
                # rendered_image = render_trimesh(smpl_mesh, R, T, 'n')
                # if input_img.shape[0] < input_img.shape[1]:
                #     rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
                # else:
                #     rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]   
                # valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
                # if overlay:
                #     output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
                #     cv2.imwrite(os.path.join('{DIR}/{seq}/init_refined_smpl_iters', '%04d.png' % it), output_img)
            
            smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
            rendered_image = render_trimesh(smpl_mesh, render_R, render_T, 'n')

            if input_img.shape[0] < input_img.shape[1]:
                rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
            else:
                rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]   
            valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
            if overlay:
                output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_smpl', '%04d.png' % idx), output_img)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_refined_mask', '%04d.png' % idx), valid_mask*255)
            last_pose.pop(0)
            last_pose.append(opt_pose.detach().clone())
            smpl_dict = {}
            smpl_dict['pose'] = opt_pose.data.squeeze().cpu().numpy()
            smpl_dict['trans'] = opt_trans.data.squeeze().cpu().numpy()
            smpl_dict['shape'] = opt_betas.data.squeeze().cpu().numpy()

            mean_shape.append(smpl_dict['shape'])
            pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_refined_smpl_files', '%04d.pkl' % idx), 'wb'))

        mean_shape = np.array(mean_shape) # can only include the shape where the poses are correct
        np.save(f'{DIR}/{seq}/mean_shape.npy', mean_shape.mean(0))

    if smooth:
        if not os.path.exists(f'{DIR}/{seq}/init_smoothed_smpl_files'):
            os.makedirs(f'{DIR}/{seq}/init_smoothed_smpl_files')
        if not os.path.exists(f'{DIR}/{seq}/init_smoothed_smpl'):
            os.makedirs(f'{DIR}/{seq}/init_smoothed_smpl')
        if not os.path.exists(f'{DIR}/{seq}/init_smoothed_mask'):
            os.makedirs(f'{DIR}/{seq}/init_smoothed_mask')
        from scipy.spatial.transform import Rotation as R
        results_p = []
        results_t = []
        images = []
        init_smpl_paths = sorted(glob.glob(f'{DIR}/{seq}/init_refined_smpl_files/*.pkl'))
        mean_sahpe = np.load(f'{DIR}/{seq}/mean_shape.npy')
        for frame, init_smpl_path in enumerate(init_smpl_paths):
            img_path = img_paths[frame]
            input_img = cv2.imread(img_path)
            init_smpl_dict = pkl.load(open(init_smpl_path, 'rb'))
            pose = init_smpl_dict['pose']
            trans = init_smpl_dict['trans']
            results_p.append(pose)
            results_t.append(trans)
            images.append(input_img)
            if frame == 2:
                r_0 = R.from_rotvec(results_p[0].reshape((24, 3)))
                r_1 = R.from_rotvec(results_p[1].reshape((24, 3)))
                r_2 = R.from_rotvec(results_p[2].reshape((24, 3)))
                avg_thetas = R.from_quat((r_1.as_quat() + r_2.as_quat()) * 0.25 + r_0.as_quat() * 0.5).as_rotvec().reshape((-1))
                avg_trans = (results_t[1] + results_t[2]) * 0.25 + results_t[0] * 0.5 # (results_t[0] + results_t[1] + results_t[2]) / 3.
                
                smpl_dict = {}
                smpl_dict['pose'] = avg_thetas
                smpl_dict['trans'] = avg_trans

                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl_files', '%04d.pkl' % (frame-2)), 'wb'))

                smpl_output = smpl_model(betas=torch.tensor(mean_sahpe)[None].cuda().float(),
                                         body_pose=torch.tensor(smpl_dict['pose'][3:])[None].cuda().float(),
                                         global_orient=torch.tensor(smpl_dict['pose'][:3])[None].cuda().float(),
                                         transl=torch.tensor(smpl_dict['trans'])[None].cuda().float())
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
                rendered_image = render_trimesh(smpl_mesh, render_R, render_T, 'n')

                if input_img.shape[0] < input_img.shape[1]:
                    rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
                else:
                    rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]   
                valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
                output_img = (rendered_image[:,:,:-1] * valid_mask + images[0] * (1 - valid_mask)).astype(np.uint8)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl', '%04d.png' % (frame-2)), output_img)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_mask', '%04d.png' % (frame-2)), valid_mask*255)
                
                avg_thetas = R.from_quat((r_0.as_quat() + r_2.as_quat()) * 0.25 + r_1.as_quat() * 0.5).as_rotvec().reshape((-1))
                avg_trans = (results_t[0] + results_t[2]) * 0.25 + results_t[1] * 0.5
                
                smpl_dict = {}
                smpl_dict['pose'] = avg_thetas
                smpl_dict['trans'] = avg_trans

                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl_files', '%04d.pkl' % (frame-1)), 'wb'))
                
                smpl_output = smpl_model(betas=torch.tensor(mean_sahpe)[None].cuda().float(),
                                         body_pose=torch.tensor(smpl_dict['pose'][3:])[None].cuda().float(),
                                         global_orient=torch.tensor(smpl_dict['pose'][:3])[None].cuda().float(),
                                         transl=torch.tensor(smpl_dict['trans'])[None].cuda().float())
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
                rendered_image = render_trimesh(smpl_mesh, render_R, render_T, 'n')

                if input_img.shape[0] < input_img.shape[1]:
                    rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
                else:
                    rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]   
                valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
                output_img = (rendered_image[:,:,:-1] * valid_mask + images[1] * (1 - valid_mask)).astype(np.uint8)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl', '%04d.png' % (frame-1)), output_img)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_mask', '%04d.png' % (frame-1)), valid_mask*255)
                
            elif frame == len(init_smpl_paths) - 1:
                r_2 = R.from_rotvec(results_p[2].reshape((24, 3)))
                r_3 = R.from_rotvec(results_p[3].reshape((24, 3)))
                r_4 = R.from_rotvec(results_p[4].reshape((24, 3)))
                
                avg_thetas = R.from_quat((r_2.as_quat() + r_4.as_quat()) * 0.25 + r_3.as_quat() * 0.5).as_rotvec().reshape((-1))
                avg_trans = (results_t[2] + results_t[4]) * 0.25 + results_t[3] * 0.5 # (results_t[2] + results_t[3] + results_t[4]) / 3.
                
                smpl_dict = {}
                smpl_dict['pose'] = avg_thetas
                smpl_dict['trans'] = avg_trans
                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl_files', '%04d.pkl' % (frame-1)), 'wb'))

                smpl_output = smpl_model(betas=torch.tensor(mean_sahpe)[None].cuda().float(),
                                         body_pose=torch.tensor(smpl_dict['pose'][3:])[None].cuda().float(),
                                         global_orient=torch.tensor(smpl_dict['pose'][:3])[None].cuda().float(),
                                         transl=torch.tensor(smpl_dict['trans'])[None].cuda().float())
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
                rendered_image = render_trimesh(smpl_mesh, render_R, render_T, 'n')

                if input_img.shape[0] < input_img.shape[1]:
                    rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
                else:
                    rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]   
                valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
                output_img = (rendered_image[:,:,:-1] * valid_mask + images[1] * (1 - valid_mask)).astype(np.uint8)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl', '%04d.png' % (frame-1)), output_img)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_mask', '%04d.png' % (frame-1)), valid_mask*255)

                avg_thetas = R.from_quat((r_2.as_quat() + r_3.as_quat()) * 0.25 + r_4.as_quat() * 0.5).as_rotvec().reshape((-1))
                avg_trans = (results_t[2] + results_t[3]) * 0.25 + results_t[4] * 0.5
                smpl_dict = {}
                smpl_dict['pose'] = avg_thetas
                smpl_dict['trans'] = avg_trans
                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl_files', '%04d.pkl' % (frame)), 'wb'))

                smpl_output = smpl_model(betas=torch.tensor(mean_sahpe)[None].cuda().float(),
                                         body_pose=torch.tensor(smpl_dict['pose'][3:])[None].cuda().float(),
                                         global_orient=torch.tensor(smpl_dict['pose'][:3])[None].cuda().float(),
                                         transl=torch.tensor(smpl_dict['trans'])[None].cuda().float())
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
                rendered_image = render_trimesh(smpl_mesh, render_R, render_T, 'n')

                if input_img.shape[0] < input_img.shape[1]:
                    rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
                else:
                    rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]   
                valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
                output_img = (rendered_image[:,:,:-1] * valid_mask + images[1] * (1 - valid_mask)).astype(np.uint8)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl', '%04d.png' % (frame)), output_img)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_mask', '%04d.png' % (frame)), valid_mask*255)
            
            if len(results_p) == 5:
                r_0 = R.from_rotvec(results_p[0].reshape((24, 3)))
                r_1 = R.from_rotvec(results_p[1].reshape((24, 3)))
                r_2 = R.from_rotvec(results_p[2].reshape((24, 3)))
                r_3 = R.from_rotvec(results_p[3].reshape((24, 3)))
                r_4 = R.from_rotvec(results_p[4].reshape((24, 3)))
                avg_thetas = R.from_quat((r_0.as_quat() + r_1.as_quat() + r_3.as_quat() + r_4.as_quat()) * 0.1 + r_2.as_quat() * 0.6).as_rotvec().reshape((-1))
                avg_trans = (results_t[0] + results_t[1] + results_t[3] + results_t[4]) * 0.1 + results_t[2] * 0.6
                smpl_dict = {}
                smpl_dict['pose'] = avg_thetas
                smpl_dict['trans'] = avg_trans
                pkl.dump(smpl_dict, open(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl_files', '%04d.pkl' % (frame-2)), 'wb'))

                smpl_output = smpl_model(betas=torch.tensor(mean_sahpe)[None].cuda().float(),
                                         body_pose=torch.tensor(smpl_dict['pose'][3:])[None].cuda().float(),
                                         global_orient=torch.tensor(smpl_dict['pose'][:3])[None].cuda().float(),
                                         transl=torch.tensor(smpl_dict['trans'])[None].cuda().float())
                smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
                smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
                rendered_image = render_trimesh(smpl_mesh, render_R, render_T, 'n')

                if input_img.shape[0] < input_img.shape[1]:
                    rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...] 
                else:
                    rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]   
                valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
                output_img = (rendered_image[:,:,:-1] * valid_mask + images[1] * (1 - valid_mask)).astype(np.uint8)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_smpl', '%04d.png' % (frame-2)), output_img)
                cv2.imwrite(os.path.join(f'{DIR}/{seq}/init_smoothed_mask', '%04d.png' % (frame-2)), valid_mask*255)

                results_p.pop(0)
                results_t.pop(0)
                images.pop(0)