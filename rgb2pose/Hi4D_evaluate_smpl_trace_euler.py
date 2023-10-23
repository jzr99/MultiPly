from youtube_pose_refinement import Renderer
import numpy as np
import torch
import cv2
import glob
import trimesh
import os
from tqdm import trange
from smplx import SMPL
import pickle as pkl
def render_trimesh(mesh,R,T, mode='np'):
    
    verts = torch.tensor(mesh.vertices).cuda().float()[None]
    faces = torch.tensor(mesh.faces).cuda()[None]
    colors = torch.tensor(mesh.visual.vertex_colors).float().cuda()[None,...,:3]/255
    renderer.set_camera(R,T)
    image = renderer.render_mesh_recon(verts, faces, colors=colors, mode=mode)[0]
    image = (255*image).data.cpu().numpy().astype(np.uint8)
    
    return image

# def transform_smpl_remain_extrinsic(curr_extrinsic, target_extrinsic, smpl_pose, smpl_trans, T_hip):
#     R_root = cv2.Rodrigues(smpl_pose[:3])[0]
#     transf_global_ori = np.linalg.inv(target_extrinsic[:3, :3]) @ curr_extrinsic[:3, :3] @ R_root
#
#     target_extrinsic[:3, -1] = curr_extrinsic[:3, :3] @ (smpl_trans + T_hip) + curr_extrinsic[:3, -1] - smpl_trans - target_extrinsic[:3, :3] @ T_hip
#
#     smpl_pose[:3] = cv2.Rodrigues(transf_global_ori)[0].reshape(3)
#     smpl_trans = np.linalg.inv(target_extrinsic[:3, :3]) @ smpl_trans  # we assume
#
#     smpl_trans = smpl_trans + (np.linalg.inv(target_extrinsic[:3, :3]) @ target_extrinsic[:3, -1])
#     target_extrinsic[:3, -1] = np.zeros_like(target_extrinsic[:3, -1])
#
#     return target_extrinsic, smpl_pose, smpl_trans

SMPL_JOINTS = 24

def compute_similarity_transform(S1, S2, num_joints, verts=None):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        if verts is not None:
            verts = verts.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # Use only body joints for procrustes
    S1_p = S1[:, :num_joints]
    S2_p = S2[:, :num_joints]
    # 1. Remove mean.
    mu1 = S1_p.mean(axis=1, keepdims=True)
    mu2 = S2_p.mean(axis=1, keepdims=True)
    X1 = S1_p - mu1
    X2 = S2_p - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if verts is not None:
        verts_hat = scale * R.dot(verts) + t
        if transposed:
            verts_hat = verts_hat.T

    if transposed:
        S1_hat = S1_hat.T

    procrustes_params = {'scale': scale,
                         'R': R,
                         'trans': t}

    if verts is not None:
        return S1_hat, verts_hat, procrustes_params
    else:
        return S1_hat, procrustes_params


def align_by_pelvis(joints, verts=None):

    left_id = 1
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    if verts is not None:
        return verts - np.expand_dims(pelvis, axis=0)
    else:
        return joints - np.expand_dims(pelvis, axis=0)

def compute_errors_joints(gt_joints, pred_joints):

    # num_joints = gt_joints[0].shape[0]
    errors, errors_verts, errors_procrustes, errors_procrustes_verts = [], [], [], []

    for i, (gt3d, pred) in enumerate(zip(gt_joints, pred_joints)):
        # Get corresponding ground truth and predicted 3d joints and verts
        gt3d = gt3d.reshape(-1, 3)

        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        # import ipdb; ipdb.set_trace()
        # Calculate joints and verts pelvis align error
        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        errors.append(np.mean(joint_error))


        # Get procrustes align error. # Not used anymore
        pred3d_sym, procrustesParam = compute_similarity_transform(pred3d, gt3d, SMPL_JOINTS)

        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        errors_procrustes.append(np.mean(pa_error))

    return errors, errors_verts, errors_procrustes, errors_procrustes_verts

def word2cam(data_world, proj_matrix):
    """
    Project 3d world coordinates to 3d camera coordinates
    data_world: (B, N, 3)
    proj_matrix: (3, 4)

    return
    data_cam: (B, N, 3)
    """

    ext_arr = np.ones((data_world.shape[0], data_world.shape[1], 1))
    data_homo = np.concatenate((data_world, ext_arr), axis=2)
    data_cam = data_homo @ proj_matrix.T

    return data_cam


DATA_DIR = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data"
DIR = '/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D'
GT_DIR = "/cluster/project/infk/hilliges/jiangze/ROMP/global_scratch/ROMP/ROMP/dataset/Hi4D/Hi4D_all/Hi4D"
RAW_DIR = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/preprocessing/raw_data"
test_conf = [
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose',
    #  'checkpoint_version': 'last.ckpt',
    # },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_meshdepth',
    #  'checkpoint_version': 'epoch=1699-loss=0.011777706444263458.ckpt',
    # },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose_smpl_surface_earlystop_certain',
    #  'checkpoint_version': 'epoch=2299-loss=0.009943617507815361.ckpt',
    # },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt',
    #  'checkpoint_version': 'epoch=0049-loss=0.045697182416915894.ckpt',
    # },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc',
    #  'checkpoint_version': 'epoch=1199-loss=0.012088967487215996.ckpt',
    # },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_surface2_depth_samGT_personid',
    #  'checkpoint_version': 'epoch=1849-loss=0.010912423022091389.ckpt',
    # },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane',
    #  'checkpoint_version': 'epoch=1799-loss=0.011541222222149372.ckpt',
    # },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_silhouette',
    #  'checkpoint_version': 'epoch=0699-loss=1.618367075920105.ckpt',
    # },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose',
    #  'checkpoint_version': 'epoch=1299-loss=0.01374354213476181.ckpt',
    #  },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_nodelay',
    #  'checkpoint_version': 'epoch=1299-loss=0.012814070098102093.ckpt',
    #  },
    # {'pair': "pair19",
    #  'action': "piggyback19",
    #  'cam_view': 4,
    #  'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
    #  'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_silhouette',
    #  'checkpoint_version': 'epoch=1299-loss=0.019231796264648438.ckpt',
    #  },
# {'pair': "pair19",
#      'action': "piggyback19",
#      'cam_view': 4,
#      'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
#      'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter',
#      'checkpoint_version': 'epoch=0499-loss=0.017574748024344444.ckpt',
#      },
#     {'pair': "pair19",
#      'action': "piggyback19",
#      'cam_view': 4,
#      'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
#      'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose',
#      'checkpoint_version': 'epoch=1849-loss=0.012226337566971779.ckpt',
#      },
#     {'pair': "pair19",
#      'action': "piggyback19",
#      'cam_view': 4,
#      'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
#      'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_loop',
#      'checkpoint_version': 'epoch=0499-loss=0.011443235911428928.ckpt',
#      },
# {'pair': "pair19",
#      'action': "piggyback19",
#      'cam_view': 4,
#      'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
#      'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_joint',
#      'checkpoint_version': 'epoch=0499-loss=0.011236797086894512.ckpt',
#      },
# {'pair': "pair19",
#      'action': "piggyback19",
#      'cam_view': 4,
#      'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
#      'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_loop_50_5',
#      'checkpoint_version': 'epoch=0499-loss=0.017181260511279106.ckpt',
#      },
# {'pair': "pair19",
#      'action': "piggyback19",
#      'cam_view': 4,
#      'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
#      'seq': 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_loop_100_10',
#      'checkpoint_version': 'epoch=0499-loss=0.03652357682585716.ckpt',
#      },
# {'pair': "pair16",
#      'action': "jump16",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair16_jump16_4',
#      'seq': 'Hi4D_pair16_jump16_4_loop',
#      'checkpoint_version': 'epoch=1499-loss=0.014593780040740967.ckpt',
#      },
# {'pair': "pair16",
#      'action': "jump16",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair16_jump16_4',
#      'seq': 'Hi4D_pair16_jump16_4_sam_loop',
#      'checkpoint_version': 'epoch=1499-loss=0.013870456255972385.ckpt',
#      },
# {'pair': "pair16",
#      'action': "jump16",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair16_jump16_4',
#      'seq': 'Hi4D_pair16_jump16_4_sam_depth_loop',
#      'checkpoint_version': 'epoch=1449-loss=0.014233583584427834.ckpt',
#      },
# {'pair': "pair16",
#      'action': "jump16",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair16_jump16_4',
#      'seq': 'Hi4D_pair16_jump16_4_sam_delay_depth_loop',
#      'checkpoint_version': 'epoch=1499-loss=0.01560750138014555.ckpt',
#      },
# {'pair': "pair15",
#      'action': "fight15",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair15_fight15_4',
#      'seq': 'Hi4D_pair15_fight15_4_loop',
#      'checkpoint_version': 'epoch=1149-loss=0.014968049712479115.ckpt',
#      },
# {'pair': "pair15",
#      'action': "fight15",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair15_fight15_4',
#      'seq': 'Hi4D_pair15_fight15_4_sam_loop',
#      'checkpoint_version': 'epoch=1149-loss=0.010965314693748951.ckpt',
#      },
# {'pair': "pair15",
#      'action': "fight15",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair15_fight15_4',
#      'seq': 'Hi4D_pair15_fight15_4_sam_depth_loop',
#      'checkpoint_version': 'epoch=1149-loss=0.01198616810142994.ckpt',
#      },
# {'pair': "pair15",
#      'action': "fight15",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair15_fight15_4',
#      'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop',
#      'checkpoint_version': 'epoch=1149-loss=0.012045151554048061.ckpt',
#      },
# {'pair': "pair15",
#      'action': "fight15",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair15_fight15_4_custom',
#      'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#      'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#      },
# {'pair': "pair15",
#      'action': "fight15",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair15_fight15_4_custom',
#      'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth10_loop_align',
#      'checkpoint_version': 'epoch=0449-loss=0.016567431390285492.ckpt',
#      },
# {'pair': "pair17",
#      'action': "dance17",
#      'cam_view': 28,
#      'inverse': False,
#      'data_seq': 'pair17_dance17_28',
#      'seq': 'Hi4D_pair17_dance17_28_loop',
#      'checkpoint_version': 'last.ckpt',
#      },
# {'pair': "pair17",
#      'action': "dance17",
#      'cam_view': 28,
#      'inverse': False,
#      'data_seq': 'pair17_dance17_28',
#      'seq': 'Hi4D_pair17_dance17_28_sam_loop',
#      'checkpoint_version': 'last.ckpt',
#      },
# {'pair': "pair17",
#      'action': "dance17",
#      'cam_view': 28,
#      'inverse': False,
#      'data_seq': 'pair17_dance17_28',
#      'seq': 'Hi4D_pair17_dance17_28_sam_depth_loop',
#      'checkpoint_version': 'last.ckpt',
#      },
# {'pair': "pair17",
#      'action': "dance17",
#      'cam_view': 28,
#      'inverse': False,
#      'data_seq': 'pair17_dance17_28',
#      'seq': 'Hi4D_pair17_dance17_28_sam_delay_depth_loop',
#      'checkpoint_version': 'last.ckpt',
#      },
# {'pair': "pair18",
#      'action': "basketball18",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair18_basketball18_4',
#      'seq': 'Hi4D_pair18_basketball18_4_loop',
#      'checkpoint_version': 'last.ckpt',
#      },
# {'pair': "pair18",
#      'action': "basketball18",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair18_basketball18_4',
#      'seq': 'Hi4D_pair18_basketball18_4_sam_loop',
#      'checkpoint_version': 'last.ckpt',
#      },
# {'pair': "pair18",
#      'action': "basketball18",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair18_basketball18_4',
#      'seq': 'Hi4D_pair18_basketball18_4_sam_depth_loop',
#      'checkpoint_version': 'last.ckpt',
#      },
# {'pair': "pair18",
#      'action': "basketball18",
#      'cam_view': 4,
#      'inverse': False,
#      'data_seq': 'pair18_basketball18_4',
#      'seq': 'Hi4D_pair18_basketball18_4_sam_delay_depth_loop',
#      'checkpoint_version': 'last.ckpt',
#      },

#     {'pair': "pair15",
#          'action': "fight15",
#          'cam_view': 4,
#          'inverse': False,
#         'raw_data': 'pair15_fight15_4_custom',
#          'data_seq': 'pair15_fight15_4_custom',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair18",
#          'action': "basketball18",
#          'cam_view': 4,
#          'inverse': False,
#         'raw_data': 'pair18_basketball18_4',
#          'data_seq': 'pair18_basketball18_4',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair16",
#          'action': "jump16",
#          'cam_view': 4,
#          'inverse': False,
#         'raw_data': 'pair16_jump16_4',
#          'data_seq': 'pair16_jump16_4',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair17",
#          'action': "dance17",
#          'cam_view': 28,
#          'inverse': False,
#         'raw_data': 'pair17_dance17_28',
#          'data_seq': 'pair17_dance17_28',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair19",
#          'action': "piggyback19",
#          'cam_view': 4,
#          'inverse': False,
#         'raw_data': 'pair19_piggyback19_4',
#          'data_seq': 'pair19_piggyback19_4',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair19",
#          'action': "piggyback19",
#          'cam_view': 4,
#          'inverse': False,
#         'raw_data': 'pair19_piggyback19_custom_vitpose_4',
#          'data_seq': 'pair19_piggyback19_custom_vitpose_4',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair19",
#          'action': "piggyback19",
#          'cam_view': 4,
#          'inverse': True,
#         'raw_data': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
#          'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair19",
#          'action': "piggyback19",
#          'cam_view': 4,
#          'inverse': True,
#         'raw_data': 'Hi4D_pair19_piggyback19_cam4_raw_openpose_test',
#          'data_seq': 'Hi4D_pair19_piggyback19_cam4_raw_openpose',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair16",
#          'action': "jump16",
#          'cam_view': 4,
#          'inverse': False,
#         'raw_data': 'pair16_jump16_vitpose_4',
#          'data_seq': 'pair16_jump16_vitpose_4',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },

# {'pair': "pair15",
#          'action': "fight15",
#          'cam_view': 4,
#          'inverse': False,
#         'raw_data': 'pair15_fight15_vitpose_4',
#          'data_seq': 'pair15_fight15_vitpose_4',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair16",
#          'action': "jump16",
#          'cam_view': 4,
#          'inverse': False,
#         'raw_data': 'pair16_jump16_vitpose_4',
#          'data_seq': 'pair16_jump16_vitpose_4',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
# {'pair': "pair17",
#          'action': "dance17",
#          'cam_view': 28,
#          'inverse': False,
#         'raw_data': 'pair17_dance17_vitpose_28',
#          'data_seq': 'pair17_dance17_vitpose_28',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
{'pair': "pair18",
         'action': "basketball18",
         'cam_view': 4,
         'inverse': False,
        'raw_data': 'pair18_basketball18_vitpose_4',
         'data_seq': 'pair18_basketball18_vitpose_4',
         # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
         # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
    },
# {'pair': "pair19",
#          'action': "piggyback19",
#          'cam_view': 4,
#          'inverse': False,
#         'raw_data': 'pair19_piggyback19_vitpose_4',
#          'data_seq': 'pair19_piggyback19_vitpose_4',
#          # 'seq': 'Hi4D_pair15_fight15_4_sam_delay_depth_loop_align',
#          # 'checkpoint_version': 'epoch=1049-loss=0.016193989664316177.ckpt',
#     },
]
for conf in test_conf:
    pair = conf['pair']
    action = conf['action']
    # seq = conf['seq']
    data_seq = conf['data_seq']
    # checkpoint_version = conf['checkpoint_version']
    cam_view = conf['cam_view']
    raw_data = conf['raw_data']
    camera_path = f"{GT_DIR}/{pair}/{action}/cameras/rgb_cameras.npz"
    Hi4D_gt_smpl_path = f"{GT_DIR}/{pair}/{action}/smpl"

    # refined_smpl_dir = f'{RAW_DIR}/{raw_data}/init_smpl_files'
    refined_smpl_dir = f'{RAW_DIR}/{raw_data}/init_refined_smpl_files'
    refined_smpl_paths = sorted(glob.glob(f"{refined_smpl_dir}/*.pkl"))

    # seq = 'courtyard_shakeHands_00_no_pose_condition_interpenetration_loss'
    # seq = 'Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose'
    # seq = 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane'
    # seq = 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_meshdepth'
    # seq = 'Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth'
    # seq = 'courtyard_shakeHands_00_loop'
    # seq = 'courtyard_shakeHands_00'
    # data_seq = 'Hi4D_pair19_piggyback19_cam4_raw_openpose' # for this data person_id 0 means person 2, and prson_id 1 means persons 1
    # data_seq = 'courtyard_shakeHands_00'
    # checkpoint_version = 'epoch=0499-loss=0.03910435736179352.ckpt'
    # checkpoint_version = 'last.ckpt'
    # checkpoint_version = 'epoch=1249-loss=0.009884528815746307.ckpt'
    # checkpoint_version = 'epoch=1699-loss=0.011777706444263458.ckpt'
    # checkpoint_version = 'epoch=0849-loss=0.11149047315120697.ckpt'
    # checkpoint_version = 'epoch=0049-loss=0.04067659005522728.ckpt'

    device = torch.device("cuda:0")
    gt_smpl_paths = sorted(glob.glob(f"{Hi4D_gt_smpl_path}/*.npz"))
    cameras = dict(np.load(camera_path))
    c = int(np.where(cameras['ids'] == cam_view)[0])
    gt_cam_intrinsics = cameras['intrinsics'][c]
    gt_cam_extrinsics = cameras['extrinsics'][c]
    try:
        inverse = conf['inverse']
    except:
        inverse = True
    if inverse:
        person_id_list = [1, 0]
    else:
        person_id_list = [0, 1]

    # person_id = 1
    # gender = 'male'
    if not os.path.exists(f'{RAW_DIR}/{raw_data}/joint_opt_smpl'):
        os.makedirs(f'{RAW_DIR}/{raw_data}/joint_opt_smpl')
    # checkpoint_path = sorted(glob.glob(f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}_wo_disp_freeze_20_every_20_opt_pose/checkpoints/*.ckpt'))[-1]
    # checkpoint_path = f"{DIR}/{seq}/checkpoints/{checkpoint_version}"
    # checkpoint = torch.load(checkpoint_path)
    # # import ipdb;ipdb.set_trace()
    # betas_0 = checkpoint['state_dict']['body_model_list.0.betas.weight']
    # betas_1 = checkpoint['state_dict']['body_model_list.1.betas.weight']
    # betas = torch.cat([betas_0, betas_1], dim=0)
    #
    # global_orient_0 = checkpoint['state_dict']['body_model_list.0.global_orient.weight']
    # global_orient_1 = checkpoint['state_dict']['body_model_list.1.global_orient.weight']
    # global_orient = torch.stack([global_orient_0, global_orient_1], dim=1)
    #
    # transl_0 = checkpoint['state_dict']['body_model_list.0.transl.weight']
    # transl_1 = checkpoint['state_dict']['body_model_list.1.transl.weight']
    # transl = torch.stack([transl_0, transl_1], dim=1)
    #
    # body_pose_0 = checkpoint['state_dict']['body_model_list.0.body_pose.weight']
    # body_pose_1 = checkpoint['state_dict']['body_model_list.1.body_pose.weight']
    # body_pose = torch.stack([body_pose_0, body_pose_1], dim=1)
    #
    #
    #
    # np.save(os.path.join(DIR, seq, 'joint_opt_smpl', 'mean_shape.npy'), betas.detach().cpu().numpy())
    # np.save(os.path.join(DIR, seq, 'joint_opt_smpl', 'poses.npy'), torch.cat((global_orient, body_pose), dim=2).detach().cpu().numpy())
    # np.save(os.path.join(DIR, seq,'joint_opt_smpl',  'normalize_trans.npy'), transl.detach().cpu().numpy())

    # camPs = np.load(f'{DATA_DIR}/{data_seq}/cameras.npz')
    gender_list = np.load(f'{DATA_DIR}/{data_seq}/gender.npy')
    pj2d_org = []
    joints = []
    cam_trans = []
    verts = []

    for person_id in person_id_list:
        smpl_model = SMPL('/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/smpl', gender=gender_list[person_id]).to(device)

        img_dir = f'{DATA_DIR}/{data_seq}/image'
        img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
        input_img = cv2.imread(img_paths[0])
        focal_length = max(input_img.shape[0], input_img.shape[1])
        cam_intrinsics = np.array([[focal_length, 0., input_img.shape[1] // 2],
                                   [0., focal_length, input_img.shape[0] // 2],
                                   [0., 0., 1.]])
        cam_extrinsics = np.eye(4)

        # temp_camP = camPs['cam_0']
        # out = cv2.decomposeProjectionMatrix(temp_camP[:3, :])
        # cam_intrinsics = out[0]

        # import ipdb;ipdb.set_trace()

        renderer = Renderer(img_size = [input_img.shape[0], input_img.shape[1]], cam_intrinsic=cam_intrinsics)
        pj2d_org_person = []
        joints_person = []
        cam_trans_person = []
        verts_person = []
        keypoint_list = []
        errors,errors_procrustes = [], []

        for i in trange(len(refined_smpl_paths)):

            seq_file = pkl.load(open(refined_smpl_paths[i], 'rb'))
            smpl_shape = np.load(f'{RAW_DIR}/{raw_data}/mean_shape.npy')[person_id]
            smpl_pose = seq_file['pose'][person_id]
            smpl_trans = seq_file['trans'][person_id]
            K = np.eye(4)
            K[:3, :3] = cam_intrinsics
            # TODO here I did not use scale factor for K
            K[0, 0] = K[0, 0]
            K[1, 1] = K[1, 1]
            K[0, 2] = K[0, 2]
            K[1, 2] = K[1, 2]
            P = K @ cam_extrinsics


            gt_smpl = np.load(gt_smpl_paths[i], allow_pickle=True)
            gt_keypoint_3d = word2cam(gt_smpl["joints_3d"], gt_cam_extrinsics)
            if inverse:
                gt_keypoint_3d = gt_keypoint_3d[1-person_id]
            else:
                gt_keypoint_3d = gt_keypoint_3d[person_id]
            # print("gt_3d.shape", gt_keypoint_3d.shape)

            input_img = cv2.imread(img_paths[i])

            out = cv2.decomposeProjectionMatrix(P[:3, :])
            # out = cv2.decomposeProjectionMatrix(camPs[f'cam_{i}'][:3, :])
            render_R = out[1]
            cam_center = out[2]
            cam_center = (cam_center[:3] / cam_center[3])[:, 0]
            render_T = -render_R @ cam_center
            current_extrinsic = np.eye(4)
            current_extrinsic[:3, :3] = render_R
            current_extrinsic[:3, 3] = render_T
            cam_trans_person.append(render_T)
            # current_extrinsic = torch.tensor(current_extrinsic).float().to(device)
            render_R = torch.tensor(render_R)[None].float()
            render_T = torch.tensor(render_T)[None].float()

            smpl_output = smpl_model(betas=torch.tensor(smpl_shape)[None].float().to(device),
                                     body_pose=torch.tensor(smpl_pose[3:])[None].float().to(device),
                                     global_orient=torch.tensor(smpl_pose[:3])[None].float().to(device),
                                     transl=torch.tensor(smpl_trans)[None].float().to(device))

            # smpl_output = smpl_model(betas = betas[[person_id]],
            #                         body_pose = body_pose[i:i+1,person_id],
            #                         global_orient = global_orient[i:i+1,person_id],
            #                         transl = transl[i:i+1,person_id])
            pred_joint_3d = smpl_output.joints.data.cpu().numpy().squeeze()
            pred_joint_3d = word2cam(pred_joint_3d[None], current_extrinsic[:3])
            pred_joint_3d = pred_joint_3d[:, :24]
            joints_person.append(pred_joint_3d[0])
            smpl_verts_numpy = smpl_output.vertices.data.cpu().numpy().squeeze()
            smpl_verts_numpy = word2cam(smpl_verts_numpy[None], current_extrinsic[:3])
            verts_person.append(smpl_verts_numpy[0])
            # print("pred_joint_3d.shape", pred_joint_3d.shape)

            gt3d = gt_keypoint_3d.reshape(-1, 3)
            pred = pred_joint_3d.reshape(-1, 3)

            gt3d = align_by_pelvis(gt3d)
            pred3d = align_by_pelvis(pred)

            # import ipdb; ipdb.set_trace()
            # Calculate joints and verts pelvis align error
            joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
            errors.append(np.mean(joint_error))

            # Get procrustes align error. # Not used anymore
            pred3d_sym, procrustesParam = compute_similarity_transform(pred3d, gt3d, SMPL_JOINTS)

            pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
            errors_procrustes.append(np.mean(pa_error))

            smpl_verts = smpl_output.vertices.data.cpu().numpy().squeeze()
            smpl_mesh = trimesh.Trimesh(smpl_verts, smpl_model.faces, process=False)
            rendered_image = render_trimesh(smpl_mesh, render_R, render_T, 'n')

            if input_img.shape[0] < input_img.shape[1]:
                rendered_image = rendered_image[abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2,...]
            else:
                rendered_image = rendered_image[:,abs(input_img.shape[0]-input_img.shape[1])//2:(input_img.shape[0]+input_img.shape[1])//2]

            valid_mask = (rendered_image[:,:,-1] > 0)[:, :, np.newaxis]
            output_img = (rendered_image[:,:,:-1] * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
            if not os.path.exists(f'{RAW_DIR}/{raw_data}/joint_opt_smpl/{person_id}'):
                os.makedirs(f'{RAW_DIR}/{raw_data}/joint_opt_smpl/{person_id}')
            cv2.imwrite(os.path.join(f'{RAW_DIR}/{raw_data}/joint_opt_smpl/{person_id}', '%04d.png' % i), output_img)

            # P = camPs[f'cam_{i}']
            smpl_joints = smpl_output.joints.data.cpu().numpy().squeeze()
            # print(smpl_joints.shape)
            # exit()
            smpl_joints = smpl_joints[:27] # original smpl point + nose + eyes
            pix_list = []
            pix_float_list = []
            for j in range(0, smpl_joints.shape[0]):
                padded_v = np.pad(smpl_joints[j], (0, 1), 'constant', constant_values=(0, 1))
                temp = P @ padded_v.T  # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
                pix = (temp / temp[2])[:2]
                output_img = cv2.circle(input_img, tuple(pix.astype(np.int32)), 3, (0, 255, 255), -1)
                pix_list.append(pix.astype(np.int32))
                pix_float_list.append(pix)
            pix_tensor = np.stack(pix_list, axis=0)
            pix_float_list_tensor = np.stack(pix_float_list, axis=0)
            keypoint_list.append(pix_tensor)
            pj2d_org_person.append(pix_float_list_tensor)
            if not os.path.exists(f'{RAW_DIR}/{raw_data}/joint_opt_smpl_joint/{person_id}'):
                os.makedirs(f'{RAW_DIR}/{raw_data}/joint_opt_smpl_joint/{person_id}')
            cv2.imwrite(os.path.join(f'{RAW_DIR}/{raw_data}/joint_opt_smpl_joint/{person_id}', '%04d.png' % i), output_img)
        pj2d_org.append(np.stack(pj2d_org_person, axis=0))
        joints.append(np.stack(joints_person, axis=0))
        verts.append(np.stack(verts_person, axis=0))
        cam_trans.append(np.stack(cam_trans_person, axis=0))
        print(f"#########################{person_id}########################{raw_data}####")
        print("MPJPE: ", np.mean(errors))
        print("PA_MPJPEL: ", np.mean(errors_procrustes))
        np.save(f'{RAW_DIR}/{raw_data}/joint_opt_smpl_joint/{person_id}.npy', np.stack(keypoint_list, axis=0))
        print(np.stack(keypoint_list, axis=0).shape)
    np.savez(f'{RAW_DIR}/{raw_data}/joint_opt_smpl_joint/all', pj2d_org=np.array(pj2d_org), joints=np.array(joints), verts=np.array(verts), cam_trans=np.array(cam_trans))