import numpy as np
import os
import glob
import trimesh
import cv2
import pickle as pkl
import ipdb
from tqdm import tqdm
import torch
from smplx import SMPL
import warnings
warnings.filterwarnings("ignore")
def transform_smpl(curr_extrinsic, target_extrinsic, smpl_pose, smpl_trans, T_hip):
    
    R_root = cv2.Rodrigues(smpl_pose[:3])[0]
    transf_global_ori = np.linalg.inv(target_extrinsic[:3,:3]) @ curr_extrinsic[:3,:3] @ R_root
    
    target_extrinsic[:3, -1] = curr_extrinsic[:3,:3] @ (smpl_trans + T_hip) + curr_extrinsic[:3, -1] - smpl_trans - target_extrinsic[:3,:3] @ T_hip 

    smpl_pose[:3] = cv2.Rodrigues(transf_global_ori)[0].reshape(3)
    smpl_trans = np.linalg.inv(target_extrinsic[:3,:3]) @ smpl_trans # we assume

    return target_extrinsic, smpl_pose, smpl_trans


def transform_smpl_remain_extrinsic(curr_extrinsic, target_extrinsic, smpl_pose, smpl_trans, T_hip):
    R_root = cv2.Rodrigues(smpl_pose[:3])[0]
    transf_global_ori = np.linalg.inv(target_extrinsic[:3, :3]) @ curr_extrinsic[:3, :3] @ R_root

    target_extrinsic[:3, -1] = curr_extrinsic[:3, :3] @ (smpl_trans + T_hip) + curr_extrinsic[:3, -1] - smpl_trans - target_extrinsic[:3, :3] @ T_hip

    smpl_pose[:3] = cv2.Rodrigues(transf_global_ori)[0].reshape(3)
    smpl_trans = np.linalg.inv(target_extrinsic[:3, :3]) @ smpl_trans  # we assume

    smpl_trans = smpl_trans + (np.linalg.inv(target_extrinsic[:3, :3]) @ target_extrinsic[:3, -1])
    target_extrinsic[:3, -1] = np.zeros_like(target_extrinsic[:3, -1])

    return target_extrinsic, smpl_pose, smpl_trans

dial_kernel = np.ones((20, 20),np.uint8)

seq = 'C7_00'
dataset = 'Hi4D' # 'youtube' # 'youtube' 'monoperfcap' # 'neuman # threedpw # ma # emdb # tiktok
transpose = True
# gender = 'm'
DIR = ''
resize_factor = 2


if dataset == 'Hi4D':
    # DIR = "/Users/jiangzeren/Downloads/eval/data/pair01/hug01"
    DIR = "/cluster/project/infk/hilliges/jiangze/ROMP/global_scratch/ROMP/ROMP/dataset/Hi4D/Hi4D_all/Hi4D/pair19/piggyback19"
    # camera_view = dict(np.load(os.path.join(DIR, "meta.npz")))["mono_cam"]
    # print("camera_view", camera_view)
    camera_view = 4
    # person_index = 0
    img_dir = f'{DIR}/images/{camera_view}'
    seq_dir = f'{DIR}/smpl'
    mask_dir_0 = f'{DIR}/seg/img_seg_mask/{camera_view}/0'
    mask_dir_1 = f'{DIR}/seg/img_seg_mask/{camera_view}/1'
    gender_list = dict(np.load(os.path.join(DIR, "meta.npz")))["genders"]

    # import pdb;pdb.set_trace()
    # save_dir = '/Users/jiangzeren/Downloads/eval/V2A_multiperson'
    save_dir = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/Hi4D_pair19_piggyback19"
    np.save(os.path.join(save_dir, "gender.npy"), gender_list)

if not os.path.exists(os.path.join(save_dir, 'image')):
    os.makedirs(os.path.join(save_dir, 'image'))
if not os.path.exists(os.path.join(save_dir, 'mask')):
    os.makedirs(os.path.join(save_dir, 'mask'))
if not os.path.exists(os.path.join(save_dir, 'mask' ,'0')):
    os.makedirs(os.path.join(save_dir, 'mask', '0'))
if not os.path.exists(os.path.join(save_dir, 'mask', '1')):
    os.makedirs(os.path.join(save_dir, 'mask', '1'))

print(img_dir)
if dataset == 'Hi4D':
    img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
else:
    img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
mask_paths_0 = sorted(glob.glob(f"{mask_dir_0}/*.png"))
mask_paths_1 = sorted(glob.glob(f"{mask_dir_1}/*.png"))
if dataset == 'Hi4D':
    seq_file_paths = sorted(glob.glob(f"{seq_dir}/*.npz"))
else:
    seq_file_paths = sorted(glob.glob(f"{seq_dir}/*.pkl"))

# if gender == 'f':
#     gender = 'female'
# elif gender == 'm':
#     gender = 'male'

# smpl_model_0 = SMPL('/Users/jiangzeren/Downloads/eval/SMPL_python_v.1.1.0/smpl/smpl', gender=gender_list[0])
smpl_model_0 = SMPL('/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/lib/smpl/smpl_model', gender=gender_list[0])
smpl_model_1 = SMPL('/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/lib/smpl/smpl_model', gender=gender_list[1])
# we use the betas from naked body not "clothed"
if dataset == "Hi4D":
    smpl_shape_0 = np.load(f"{DIR}/smpl/000006.npz", allow_pickle=True)['betas'][0]
    smpl_shape_1 = np.load(f"{DIR}/smpl/000006.npz", allow_pickle=True)['betas'][1]
    T_hip_0 = smpl_model_0.get_T_hip(betas=torch.tensor(smpl_shape_0)[None].float()).squeeze().cpu().numpy()
    T_hip_1 = smpl_model_1.get_T_hip(betas=torch.tensor(smpl_shape_1)[None].float()).squeeze().cpu().numpy()

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
elif dataset == 'emdb':
    focal_length = 960
    cam_intrinsics = np.array([[focal_length, 0., 360.],[0.,focal_length, 480.],[0.,0.,1.]])
elif dataset == 'ma':
    focal_length = 1280
    cam_intrinsics = np.array([[focal_length, 0., 360.],[0.,focal_length, 640.],[0.,0.,1.]])
elif dataset == 'tiktok':
    focal_length = 1080
    cam_intrinsics = np.array([[focal_length, 0., 302.],[0.,focal_length, 540.],[0.,0.,1.]])

cam_extrinsics = np.eye(4)

if dataset == 'Hi4D':
    cameras = dict(np.load(f'{DIR}/cameras/rgb_cameras.npz'))
    # get camera parameters
    i = int(np.where(cameras['ids'] == camera_view)[0])
    cam_intrinsics = cameras['intrinsics'][i]
    cam_extrinsics[:3,:4] = cameras['extrinsics'][i]


K = np.eye(4)
K[:3, :3] = cam_intrinsics
K[0, 0] = K[0, 0] / resize_factor
K[1, 1] = K[1, 1] / resize_factor
K[0, 2] = K[0, 2] / resize_factor
K[1, 2] = K[1, 2] / resize_factor
output_trans = []
output_pose = []
output_P = {}
max_human_sphere_all = 0
for idx, img_path in enumerate(tqdm(img_paths)):
    # resize image for speed-up
    img = cv2.imread(img_path)
    img = cv2.resize(img, (img.shape[1] // resize_factor, img.shape[0] // resize_factor))

    if dataset == "Hi4D":
        seq_file = np.load(seq_file_paths[idx], allow_pickle=True)
    else:
        seq_file = pkl.load(open(seq_file_paths[idx], 'rb'))


    mask_0 = cv2.imread(mask_paths_0[idx]) # cv2.imread(f"{mask_dir}/{frame:04d}.png")
    mask_0 = cv2.resize(mask_0, (mask_0.shape[1] // resize_factor, mask_0.shape[0] // resize_factor))
    # dilate mask
    mask_0 = cv2.dilate(mask_0, dial_kernel)
    # no need to dilate ground mask

    mask_1 = cv2.imread(mask_paths_1[idx])  # cv2.imread(f"{mask_dir}/{frame:04d}.png")
    mask_1 = cv2.resize(mask_1, (mask_1.shape[1] // resize_factor, mask_1.shape[0] // resize_factor))
    # dilate mask
    mask_1 = cv2.dilate(mask_1, dial_kernel)
    # no need to dilate ground mask

    cv2.imwrite(os.path.join(save_dir, 'image/%04d.png' % idx), img)
    cv2.imwrite(os.path.join(save_dir, 'mask/0/%04d.png' % idx), mask_0)
    cv2.imwrite(os.path.join(save_dir, 'mask/1/%04d.png' % idx), mask_1)
    # cv2.imwrite(os.path.join(save_dir, 'ground_mask/%04d.png' % idx), ground_mask)
    # cv2.imwrite(os.path.join(save_dir, 'normal/%04d.png' % idx), normal)

    if dataset =="Hi4D":
        smpl_pose_0 = np.concatenate([seq_file['global_orient'][0],seq_file['body_pose'][0]], axis=0)
        smpl_trans_0 = seq_file['transl'][0]
        smpl_pose_1 = np.concatenate([seq_file['global_orient'][1], seq_file['body_pose'][1]], axis=0)
        smpl_trans_1 = seq_file['transl'][1]
    
    # transform the spaces such that our model space is equal to the ICON (PIFuHD) model space  
    target_extrinsic = np.eye(4)
    target_extrinsic[1:3] *= -1
    # target_extrinsic, smpl_pose, smpl_trans = transform_smpl(cam_extrinsics, target_extrinsic, smpl_pose, smpl_trans, T_hip)
    target_extrinsic_0, smpl_pose_0, smpl_trans_0 = transform_smpl_remain_extrinsic(cam_extrinsics, target_extrinsic, smpl_pose_0, smpl_trans_0, T_hip_0)
    target_extrinsic_1, smpl_pose_1, smpl_trans_1 = transform_smpl_remain_extrinsic(cam_extrinsics, target_extrinsic,
                                                                                    smpl_pose_1, smpl_trans_1, T_hip_1)

    smpl_output_0 = smpl_model_0(betas=torch.tensor(smpl_shape_0)[None].float(),
                             body_pose=torch.tensor(smpl_pose_0[3:])[None].float(),
                             global_orient=torch.tensor(smpl_pose_0[:3])[None].float(),
                             transl=torch.tensor(smpl_trans_0)[None].float())
    smpl_output_1 = smpl_model_1(betas=torch.tensor(smpl_shape_1)[None].float(),
                               body_pose=torch.tensor(smpl_pose_1[3:])[None].float(),
                               global_orient=torch.tensor(smpl_pose_1[:3])[None].float(),
                               transl=torch.tensor(smpl_trans_1)[None].float())
    
    smpl_verts_0 = smpl_output_0.vertices.data.cpu().numpy().squeeze()
    smpl_verts_1 = smpl_output_1.vertices.data.cpu().numpy().squeeze()
    smpl_verts_all = np.concatenate([smpl_verts_0,smpl_verts_1],axis=0)
    # import pdb;pdb.set_trace()

    # we need to normalize the trans for every frame due to the large global movement
    v_max = smpl_verts_all.max(axis=0)
    v_min = smpl_verts_all.min(axis=0)
    normalize_shift = -(v_max + v_min) / 2.

    # check whether humans are within the sphere
    smpl_verts_all = smpl_verts_all + normalize_shift.reshape(1, 3)
    max_human_sphere = np.linalg.norm(smpl_verts_all, axis=1).max()
    if max_human_sphere > max_human_sphere_all:
        max_human_sphere_all = max_human_sphere

    trans_0 = smpl_trans_0 + normalize_shift
    trans_1 = smpl_trans_1 + normalize_shift
    
    target_extrinsic[:3, -1] = target_extrinsic[:3, -1] - (target_extrinsic[:3, :3] @ normalize_shift)

    P = K @ target_extrinsic
    output_trans.append(np.stack([trans_0, trans_1], axis=0))
    output_pose.append(np.stack([smpl_pose_0, smpl_pose_1], axis=0))
    output_P[f"cam_{idx}"] = P

print("max_human_sphere_all: ", max_human_sphere_all)
np.save(os.path.join(save_dir, 'poses.npy'), np.array(output_pose))
print('output_pose', np.array(output_pose).shape)
np.save(os.path.join(save_dir, 'mean_shape.npy'), np.stack([smpl_shape_0, smpl_shape_1], axis=0))
print('mean_shape', np.stack([smpl_shape_0, smpl_shape_1], axis=0).shape)
np.save(os.path.join(save_dir, 'normalize_trans.npy'), np.array(output_trans))
print('normalize_trans', np.array(output_trans).shape)
# np.save(os.path.join(save_dir, 'cameras.npy'), np.array(output_P))
np.savez(os.path.join(save_dir, "cameras.npz"), **output_P)

# re-project to images to debug

smpl_output_0 = smpl_model_0(betas=torch.tensor(smpl_shape_0)[None].float(),
                         body_pose=torch.tensor(smpl_pose_0[3:])[None].float(),
                         global_orient=torch.tensor(smpl_pose_0[:3])[None].float(),
                         transl=torch.tensor(trans_0)[None].float())
smpl_verts_0 = smpl_output_0.vertices.data.cpu().numpy().squeeze()
_ = trimesh.Trimesh(smpl_verts_0).export(os.path.join(save_dir, 'test_0.ply'))

for j in range(0, smpl_verts_0.shape[0]):
    padded_v = np.pad(smpl_verts_0[j], (0,1), 'constant', constant_values=(0,1))
    temp = P @ padded_v.T # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
    pix = (temp/temp[2])[:2]
    output_img = cv2.circle(img, tuple(pix.astype(np.int32)), 3, (0,255,255), -1)

cv2.imwrite(os.path.join(save_dir, 'test_0.png'), output_img)

smpl_output_1 = smpl_model_1(betas=torch.tensor(smpl_shape_1)[None].float(),
                         body_pose=torch.tensor(smpl_pose_1[3:])[None].float(),
                         global_orient=torch.tensor(smpl_pose_1[:3])[None].float(),
                         transl=torch.tensor(trans_1)[None].float())
smpl_verts_1 = smpl_output_1.vertices.data.cpu().numpy().squeeze()
_ = trimesh.Trimesh(smpl_verts_1).export(os.path.join(save_dir, 'test_1.ply'))

for j in range(0, smpl_verts_1.shape[0]):
    padded_v = np.pad(smpl_verts_1[j], (0,1), 'constant', constant_values=(0,1))
    temp = P @ padded_v.T # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
    pix = (temp/temp[2])[:2]
    output_img = cv2.circle(img, tuple(pix.astype(np.int32)), 3, (0,255,255), -1)
# ipdb.set_trace()
cv2.imwrite(os.path.join(save_dir, 'test_1.png'), output_img)

