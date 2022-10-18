import trimesh
import numpy as np
import glob
from tqdm import tqdm
from scipy.spatial import KDTree
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

def compute_iou(mesh, gt_mesh):
    mesh_bounds = mesh.bounds
    gt_mesh_bounds = gt_mesh.bounds
    xx1 = np.max([mesh_bounds[0,0], gt_mesh_bounds[0,0]])
    yy1 = np.max([mesh_bounds[0,1], gt_mesh_bounds[0,1]])
    zz1 = np.max([mesh_bounds[0,2], gt_mesh_bounds[0,2]])

    xx2 = np.min([mesh_bounds[1,0], gt_mesh_bounds[1,0]])
    yy2 = np.min([mesh_bounds[1,1], gt_mesh_bounds[1,1]])
    zz2 = np.min([mesh_bounds[1,2], gt_mesh_bounds[1,2]])

    vol1 = (mesh_bounds[1,0] - mesh_bounds[0,0]) * (mesh_bounds[1,1] - mesh_bounds[0,1]) * (mesh_bounds[1,2] - mesh_bounds[0,2])
    vol2 = (gt_mesh_bounds[1,0] - gt_mesh_bounds[0,0]) * (gt_mesh_bounds[1,1] - gt_mesh_bounds[0,1]) * (gt_mesh_bounds[1,2] - gt_mesh_bounds[0,2])
    inter_vol = np.max([0, xx2 - xx1]) * np.max([0, yy2 - yy1]) * np.max([0, zz2 - zz1])

    iou = inter_vol / (vol1 + vol2 - inter_vol + 1e-11)
    return iou

def nc_m2m(mesh_src, mesh_tgt):
    ''' Computes minimal distances of each point in points_src to points_tgt.

    Args:
        points_src (numpy array): source points
        normals_src (numpy array): source normals
        points_tgt (numpy array): target points
        normals_tgt (numpy array): target normals
    '''
    points_src, normals_src = mesh_src.vertices, mesh_src.vertex_normals
    points_tgt, normals_tgt = mesh_tgt.vertices, mesh_tgt.vertex_normals

    kdtree = KDTree(points_tgt)
    dist, idx = kdtree.query(points_src)
    # import pdb
    # pdb.set_trace()
    if normals_src is not None and normals_tgt is not None:
        normals_src = \
            normals_src / np.linalg.norm(normals_src, axis=-1, keepdims=True)
        normals_tgt = \
            normals_tgt / np.linalg.norm(normals_tgt, axis=-1, keepdims=True)

        normals_dot_product = (normals_tgt[idx] * normals_src).sum(axis=-1)
        # Handle normals that point into wrong direction gracefully
        # (mostly due to mehtod not caring about this in generation)
        normals_dot_product = np.abs(normals_dot_product)
        normals_dot_product[np.isnan(normals_dot_product)] = 1.
    else:
        normals_dot_product = np.array(
            [np.nan] * points_src.shape[0], dtype=np.float32)
    normals_dot_product = normals_dot_product.mean()

    return dist.mean(), normals_dot_product

def compute_metric(mesh, gt_mesh, num_samples=10000):
    iou = compute_iou(mesh, gt_mesh)
    chamfer_dist = dist_m2m(mesh, gt_mesh, num_samples)
    _, acc_normals = nc_m2m(mesh, gt_mesh)
    _, comp_normals = nc_m2m(gt_mesh, mesh)

    normal_consistency = (acc_normals + comp_normals) / 2.
    # output_dict = {'iou': iou, 'chamfer': chamfer_dist, "p2s": acc_dist, 'nc': normal_consistency}
    output_dict = {'iou': iou, 'chamfer': chamfer_dist, 'nc': normal_consistency} # {'iou': iou, 'chamfer': chamfer_dist, 'nc': normal_consistency}

    return output_dict

seq = '00020_Gorilla'
if seq == 'outdoors_fencing_01':
    start_idx = 0 # 546

gt_mesh_dir = f'/home/chen/disk2/RGB_PINA_MoCap/{seq}/meshes_vis'
# ours_mesh_w_ori_pose_dir = f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled'
# selfrecon_mesh_w_ori_pose_dir = f'/home/chen/SelfReconCode/data/{seq}/result/final_meshes_transformed'
ours_mesh_w_est_pose_dir = f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}_wo_disp_freeze_20_every_20_opt_pose/test_mesh_scaled'
selfrecon_mesh_w_est_pose_dir = f'/home/chen/disk2/SelfRecon_results/{seq}/result/final_meshes_transformed'
icon_mesh_w_est_pose_dir = f'/home/chen/disk2/ICON_new_results/{seq}/icon-filter/test_mesh'
# ours_mesh_wo_opt_smpl_dir = f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}_wo_disp_freeze_20_every_20_opt_pose_no/test_mesh_scaled'

skip = 1

gt_mesh_paths = sorted(glob.glob(f'{gt_mesh_dir}/*.obj'))[::skip]
# ours_mesh_w_ori_pose_paths = sorted(glob.glob(f'{ours_mesh_w_ori_pose_dir}/*.ply'))
# selfrecon_mesh_w_ori_pose_paths = sorted(glob.glob(f'{selfrecon_mesh_w_ori_pose_dir}/*.ply'))
ours_mesh_w_est_pose_paths = sorted(glob.glob(f'{ours_mesh_w_est_pose_dir}/*.ply'))[::skip]
selfrecon_mesh_w_est_pose_paths = sorted(glob.glob(f'{selfrecon_mesh_w_est_pose_dir}/*.ply'))[::skip]
icon_mesh_w_est_pose_paths = sorted(glob.glob(f'{icon_mesh_w_est_pose_dir}/*.obj'))[::skip]
# ours_mesh_wo_opt_smpl_paths = sorted(glob.glob(f'{ours_mesh_wo_opt_smpl_dir}/*.ply'))[::skip]

ours_cd = []
selfrecon_cd = []
icon_cd = []
ours_wo_opt_smpl_cd = []

ours_nc = []
selfrecon_nc = []
icon_nc = []
ours_wo_opt_smpl_nc = []

ours_iou = []
selfrecon_iou = []
icon_iou = []
ours_wo_opt_smpl_iou = []

assert len(gt_mesh_paths) == len(ours_mesh_w_est_pose_paths) == len(selfrecon_mesh_w_est_pose_paths) == len(icon_mesh_w_est_pose_paths)
for idx, our_mesh_w_est_pose_path in tqdm(enumerate(ours_mesh_w_est_pose_paths)):

    # load target and source meshes
    gt_mesh = trimesh.load(gt_mesh_paths[idx], process=False)
    ours_mesh_w_est_pose = trimesh.load(our_mesh_w_est_pose_path, process=False)
    selfrecon_mesh_w_est_pose = trimesh.load(selfrecon_mesh_w_est_pose_paths[idx], process=False)
    icon_mesh_w_ori_pose = trimesh.load(icon_mesh_w_est_pose_paths[idx], process=False)
    # ours_mesh_wo_opt_smpl = trimesh.load(ours_mesh_wo_opt_smpl_paths[idx], process=False)

    # print('ours:', dist_m2m(gt_mesh, ours_mesh_w_est_pose))
    # print('selfrecon:', dist_m2m(gt_mesh, selfrecon_mesh_w_est_pose))
    # print('icon:', dist_m2m(gt_mesh, icon_mesh_w_ori_pose))

    ours_dict = compute_metric(ours_mesh_w_est_pose, gt_mesh)
    selfrecon_dict = compute_metric(selfrecon_mesh_w_est_pose, gt_mesh)
    icon_dict = compute_metric(icon_mesh_w_ori_pose, gt_mesh)
    # ours_wo_opt_smpl_dict = compute_metric(ours_mesh_wo_opt_smpl, gt_mesh)
    
    ours_cd.append(ours_dict['chamfer'])
    selfrecon_cd.append(selfrecon_dict['chamfer'])
    icon_cd.append(icon_dict['chamfer'])
    # ours_wo_opt_smpl_cd.append(ours_wo_opt_smpl_dict['chamfer'])

    ours_nc.append(ours_dict['nc'])
    selfrecon_nc.append(selfrecon_dict['nc'])
    icon_nc.append(icon_dict['nc'])
    # ours_wo_opt_smpl_nc.append(ours_wo_opt_smpl_dict['nc'])

    ours_iou.append(ours_dict['iou'])
    selfrecon_iou.append(selfrecon_dict['iou'])
    icon_iou.append(icon_dict['iou'])
    # ours_wo_opt_smpl_iou.append(ours_wo_opt_smpl_dict['iou'])

    print('ours:', ours_dict)
    print('selfrecon:', selfrecon_dict)
    print('icon:', icon_dict)
    # print('ours_wo_opt_smpl:', ours_wo_opt_smpl_dict)

import ipdb
ipdb.set_trace()