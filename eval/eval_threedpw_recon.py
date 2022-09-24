import trimesh
import numpy as np
import glob

def dist_m2m(mesh_src, mesh_tgt, num_samples=10000):
    src_surf_pts, _ = trimesh.sample.sample_surface(mesh_src, num_samples)
    tgt_surf_pts, _ = trimesh.sample.sample_surface(mesh_tgt, num_samples)

    _, src_tgt_dist, _ = trimesh.proximity.closest_point(mesh_tgt, src_surf_pts)
    _, tgt_src_dist, _ = trimesh.proximity.closest_point(mesh_src, tgt_surf_pts)

    src_tgt_dist[np.isnan(src_tgt_dist)] = 0
    tgt_src_dist[np.isnan(tgt_src_dist)] = 0

    src_tgt_dist = src_tgt_dist.mean()
    tgt_src_dist = tgt_src_dist.mean()
    # L2-norm chamfer distance
    chamfer_dist = (src_tgt_dist + tgt_src_dist) / 2    

    return chamfer_dist

seq = 'outdoors_fencing_01'
if seq == 'outdoors_fencing_01':
    start_idx = 546

gt_mesh_dir = f'/home/chen/disk2/3DPW_GT/{seq}/mesh'
ours_mesh_w_ori_pose_dir = f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled'
selfrecon_mesh_w_ori_pose_dir = f'/home/chen/SelfReconCode/data/{seq}/result/final_meshes_transformed'

gt_mesh_paths = sorted(glob.glob(f'{gt_mesh_dir}/*.obj'))
ours_mesh_w_ori_pose_paths = sorted(glob.glob(f'{ours_mesh_w_ori_pose_dir}/*.ply'))
selfrecon_mesh_w_ori_pose_paths = sorted(glob.glob(f'{selfrecon_mesh_w_ori_pose_dir}/*.ply'))

ours_cd = []
selfrecon_cd = []

for idx, our_mesh_w_ori_pose_path in enumerate(ours_mesh_w_ori_pose_paths):

    gt_mesh = trimesh.load(gt_mesh_paths[idx + start_idx], process=False)
    ours_mesh_w_ori_pose = trimesh.load(our_mesh_w_ori_pose_path, process=False)
    # ours_mesh_w_opt_pose = trimesh.load('/home/chen/RGB-PINA/code/outputs/ThreeDPW/outdoors_fencing_01_wo_disp_freeze_20_every_20_opt_pose/0000_deformed_w_opt_pose.ply')
    # ours_mesh_wo_opt_pose = trimesh.load('/home/chen/RGB-PINA/code/outputs/ThreeDPW/outdoors_fencing_01_wo_disp_freeze_20_every_20/test_mesh_scaled/0000_deformed.ply')
    selfrecon_mesh_w_ori_pose = trimesh.load(selfrecon_mesh_w_ori_pose_paths[idx], process=False)
    print(dist_m2m(gt_mesh, ours_mesh_w_ori_pose))
    print(dist_m2m(gt_mesh, selfrecon_mesh_w_ori_pose))
    ours_cd.append(dist_m2m(gt_mesh, ours_mesh_w_ori_pose))
    # print(dist_m2m(gt_mesh, ours_mesh_w_opt_pose))
    # print(dist_m2m(gt_mesh, ours_mesh_wo_opt_pose))
    selfrecon_cd.append(dist_m2m(gt_mesh, selfrecon_mesh_w_ori_pose))
import ipdb
ipdb.set_trace()