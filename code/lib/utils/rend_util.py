import numpy as np
import imageio
import skimage
import cv2
import torch
from torch.nn import functional as F
import trimesh

def get_psnr(img1, img2, normalize_rgb=False):
    if normalize_rgb: # [-1,1] --> [0,1]
        img1 = (img1 + 1.) / 2.
        img2 = (img2 + 1. ) / 2.

    mse = torch.mean((img1 - img2) ** 2)
    psnr = -10. * torch.log(mse) / torch.log(torch.Tensor([10.]).cuda())

    return psnr


def load_rgb(path, normalize_rgb = False):
    img = imageio.imread(path)
    img = skimage.img_as_float32(img)

    if normalize_rgb: # [-1,1] --> [0,1]
        img -= 0.5
        img *= 2.
    img = img.transpose(2, 0, 1)
    return img


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv2.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K/K[2,2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3,3] = (t[:3] / t[3])[:,0]

    return intrinsics, pose


def get_camera_params(uv, pose, intrinsics):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:]
        R = quat_to_rot(pose[:,:4])
        p = torch.eye(4).repeat(pose.shape[0],1,1).cuda().float()
        p[:, :3, :3] = R
        p[:, :3, 3] = cam_loc
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        p = pose

    batch_size, num_samples, _ = uv.shape

    depth = torch.ones((batch_size, num_samples)).cuda()
    x_cam = uv[:, :, 0].view(batch_size, -1)
    y_cam = uv[:, :, 1].view(batch_size, -1)
    z_cam = depth.view(batch_size, -1)

    pixel_points_cam = lift(x_cam, y_cam, z_cam, intrinsics=intrinsics)

    # permute for batch matrix product
    pixel_points_cam = pixel_points_cam.permute(0, 2, 1)

    world_coords = torch.bmm(p, pixel_points_cam).permute(0, 2, 1)[:, :, :3]
    ray_dirs = world_coords - cam_loc[:, None, :]
    ray_dirs = F.normalize(ray_dirs, dim=2)

    return ray_dirs, cam_loc


def get_camera_for_plot(pose):
    if pose.shape[1] == 7: #In case of quaternion vector representation
        cam_loc = pose[:, 4:].detach()
        R = quat_to_rot(pose[:,:4].detach())
    else: # In case of pose matrix representation
        cam_loc = pose[:, :3, 3]
        R = pose[:, :3, :3]
    cam_dir = R[:, :3, 2]
    return cam_loc, cam_dir


def lift(x, y, z, intrinsics):
    # parse intrinsics
    intrinsics = intrinsics.cuda()
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]

    x_lift = (x - cx.unsqueeze(-1) + cy.unsqueeze(-1)*sk.unsqueeze(-1)/fy.unsqueeze(-1) - sk.unsqueeze(-1)*y/fy.unsqueeze(-1)) / fx.unsqueeze(-1) * z
    y_lift = (y - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z

    # homogeneous
    return torch.stack((x_lift, y_lift, z, torch.ones_like(z).cuda()), dim=-1)


def quat_to_rot(q):
    batch_size, _ = q.shape
    q = F.normalize(q, dim=1)
    R = torch.ones((batch_size, 3,3)).cuda()
    qr=q[:,0]
    qi = q[:, 1]
    qj = q[:, 2]
    qk = q[:, 3]
    R[:, 0, 0]=1-2 * (qj**2 + qk**2)
    R[:, 0, 1] = 2 * (qj *qi -qk*qr)
    R[:, 0, 2] = 2 * (qi * qk + qr * qj)
    R[:, 1, 0] = 2 * (qj * qi + qk * qr)
    R[:, 1, 1] = 1-2 * (qi**2 + qk**2)
    R[:, 1, 2] = 2*(qj*qk - qi*qr)
    R[:, 2, 0] = 2 * (qk * qi-qj * qr)
    R[:, 2, 1] = 2 * (qj*qk + qi*qr)
    R[:, 2, 2] = 1-2 * (qi**2 + qj**2)
    return R


def rot_to_quat(R):
    batch_size, _,_ = R.shape
    q = torch.ones((batch_size, 4)).cuda()

    R00 = R[:, 0,0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]

    q[:,0]=torch.sqrt(1.0+R00+R11+R22)/2
    q[:, 1]=(R21-R12)/(4*q[:,0])
    q[:, 2] = (R02 - R20) / (4 * q[:, 0])
    q[:, 3] = (R10 - R01) / (4 * q[:, 0])
    return q


def get_sphere_intersections(cam_loc, ray_directions, r = 1.0):
    # Input: n_rays x 3 ; n_rays x 3
    # Output: n_rays x 1, n_rays x 1 (close and far)

    ray_cam_dot = torch.bmm(ray_directions.view(-1, 1, 3),
                            cam_loc.view(-1, 3, 1)).squeeze(-1)
    under_sqrt = ray_cam_dot ** 2 - (cam_loc.norm(2, 1, keepdim=True) ** 2 - r ** 2)

    # sanity check
    if (under_sqrt <= 0).sum() > 0:
        print('BOUNDING SPHERE PROBLEM!')
        exit()

    sphere_intersections = torch.sqrt(under_sqrt) * torch.Tensor([-1, 1]).cuda().float() - ray_cam_dot
    sphere_intersections = sphere_intersections.clamp_min(0.0)

    return sphere_intersections

def get_smpl_intersection(cam_loc, ray_directions, smpl_mesh, interval_dist=0.1):
    # smpl mesh scaling or bounding box with scaling? 
    bbox = smpl_mesh.apply_scale(1.2).bounding_box
    n_imgs, n_pix, _ = ray_directions.shape
    # smpl_mesh.apply_scale(1.1)
    
    ray_dirs = ray_directions[0].clone().cpu().numpy()
    ray_origins = np.tile(cam_loc[0].clone().cpu().numpy(), n_pix).reshape(n_pix, 3)
    locations, index_ray, _ = bbox.ray.intersects_location(ray_origins=ray_origins, ray_directions=ray_dirs, multiple_hits=False)
    mask_intersect = np.zeros(ray_dirs.shape[0], dtype=np.bool)
    
    mask_intersect[index_ray] = True
    unfinished_mask_start = torch.from_numpy(mask_intersect).cuda()
    intersect_dis = np.linalg.norm(ray_origins[index_ray] - locations, axis=1)

    curr_start_points = torch.zeros(n_pix, 3).cuda().float()
    curr_start_points[unfinished_mask_start] = torch.tensor(locations - interval_dist * ray_dirs[mask_intersect]).cuda().float()
    acc_start_dis = torch.zeros(n_pix).cuda().float()
    acc_start_dis[unfinished_mask_start] = torch.tensor(intersect_dis - interval_dist).cuda().float()
    acc_end_dis = torch.zeros(n_pix).cuda().float()
    acc_end_dis[unfinished_mask_start] = torch.tensor(intersect_dis + interval_dist).cuda().float()

    min_dis = acc_start_dis.clone()
    max_dis = acc_end_dis.clone()
    return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

def get_bbox_intersection(cam_loc, ray_directions, smpl_mesh):
    # smpl mesh scaling or bounding box with scaling? 
    bbox = smpl_mesh.apply_scale(1.5).bounding_box

    n_pix, _ = ray_directions.shape

    ray_dirs_np = ray_directions.clone().cpu().numpy()
    ray_origins_np = np.tile(cam_loc[0].clone().cpu().numpy(), n_pix).reshape(n_pix, 3)
    locations, index_ray, _ = bbox.ray.intersects_location(ray_origins=ray_origins_np, ray_directions=ray_dirs_np, multiple_hits=True)
    # strong assumption that either the ray has zero hit or two hits!
    num_ray_hits = locations.shape[0] // 2
    if num_ray_hits == n_pix:
        dists = np.linalg.norm(ray_origins_np[index_ray] - locations, axis=1)
        # condition is that all rays intersect the bbox
        near = dists[:n_pix]
        far = dists[n_pix:]
        return torch.tensor(near).cuda().float(), torch.tensor(far).cuda().float(), None
    else:

        unhit_first_index_ray = set(np.arange(0, 512)).difference(index_ray[:num_ray_hits])
        unhit_second_index_ray = set(np.arange(0, 512)).difference(index_ray[num_ray_hits:])
        if unhit_first_index_ray != unhit_second_index_ray:
            import ipdb
            ipdb.set_trace()
        unhit_index_ray = list(unhit_first_index_ray)
        to_pad_index_ray = np.random.choice(index_ray[:num_ray_hits].shape[0], len(unhit_index_ray))
        
        near_hit_locations = np.zeros((n_pix, locations.shape[1]))
        near_hit_locations[index_ray[:num_ray_hits]] = locations[:num_ray_hits]
        # padding the invalid two hits 
        near_hit_locations[unhit_index_ray] = locations[:num_ray_hits][to_pad_index_ray]

        far_hit_locations = np.zeros((n_pix, locations.shape[1]))
        far_hit_locations[index_ray[num_ray_hits:]] = locations[num_ray_hits:]
        # padding the invalid two hits
        far_hit_locations[unhit_index_ray] = locations[num_ray_hits:][to_pad_index_ray]

        near = np.linalg.norm(ray_origins_np - near_hit_locations, axis=1)

        far = np.linalg.norm(ray_origins_np - far_hit_locations, axis=1)

        ray_dirs_np[unhit_index_ray] = ray_dirs_np[to_pad_index_ray]

        padded_ray_dirs = torch.tensor(ray_dirs_np).cuda().float()
        return torch.tensor(near).cuda().float(), torch.tensor(far).cuda().float(), padded_ray_dirs
    