import numpy as np
import glob
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score, classification_report
# EXP_NAME = "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt"
# # EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose"
# # EXP_NAME = "Hi4D_pair19_piggyback19_temporal"
# # EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto"
# GT_NAME = "pair19/piggyback19"
# CAM_NAME = "4"
# inverse = True

def show_mask(mask, ax, random_color=False, color=None):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    elif color is not None:
        color = np.array(color)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=0.5)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=0.5)




# EXP_NAME = "taichi01_sam_delay_depth_loop_2_MLP_vitpose_openpose"
# # EXP_NAME = "Hi4D_pair19_piggyback19_4_sam_delay_depth_2_noshare"
# # EXP_NAME = "Hi4D_pair17_dance17_28_sam_delay_depth_loop_0pose_vitpose_2_noshare"
# # EXP_NAME = "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_2_noshare"
# # EXP_NAME = "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2_edge_noshare"
# # EXP_NAME_test = "Hi4D_pair19_piggyback19_temporal"
# # EXP_NAME = "Hi4D_pair19_piggyback19_temporal"
# # EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto
# GT_NAME = "pair19/piggyback19"
# CAM_NAME = "4"
# inverse = False
#
# EXP_ROOT =  f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/{EXP_NAME}/stage_sam_mask/"
# EXP_DIR = f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/{EXP_NAME}/stage_sam_mask/*"
# GT_DIR = f"/cluster/project/infk/hilliges/jiangze/ROMP/global_scratch/romp_data/ROMP_datasets/Hi4D/Hi4D_all/Hi4D/{GT_NAME}/seg/img_seg_mask/{CAM_NAME}"
# sam_mask_list = sorted(glob.glob(EXP_DIR))
# sam_mask_test_list = [
#     EXP_ROOT + '00000',
#     EXP_ROOT + '01000',
#     EXP_ROOT + '02000',
# ]


# subject = 'ma_wechat1'
# seq = f'{subject}'
# result_dir = f'/home/chen/release_tmp/RGB-PINA/outputs/Video/{seq}'
# result_dir = '/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/taichi01_sam_delay_depth_loop_2_MLP_vitpose_openpose'
# data_dir = f'/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/taichi01_vitpose_openpose'
# result_dir = '/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/Hi4D_pair19_piggyback19_4_sam_delay_depth_2_noshare_fix'
# result_dir = '/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/Hi4D_pair19_piggyback19_4_sam_delay_depth_2_noshare'
# data_dir = f'/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/pair19_piggyback19_vitpose_4'

result_dir = '/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_2_noshare'
data_dir = f'/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/pair15_fight15_vitpose_4'
sam_mask_path = os.path.join(result_dir, 'stage_sam_mask/01500/sam_opt_mask.npy')
# mesh_mask_path = os.path.join(result_dir, 'stage_instance_mask/00200/all_person_smpl_mask.npy')
mesh_mask_path = os.path.join(result_dir, 'stage_instance_mask/00000/all_person_smpl_mask.npy')

soft_mask_path = sorted(glob.glob(os.path.join(result_dir, 'test_mask/-1/*.png')))

image_paths = sorted(glob.glob(os.path.join(data_dir, 'image/*.png')))

save_path_sam = os.path.join(result_dir, 'figure_sam_mask')
save_path_smpl = os.path.join(result_dir, 'figure_smpl_mask')
save_path_soft = os.path.join(result_dir, 'figure_soft_mask')
os.makedirs(save_path_sam, exist_ok=True)
os.makedirs(save_path_smpl, exist_ok=True)
os.makedirs(save_path_soft, exist_ok=True)

smpl_mask = np.load(mesh_mask_path)
sam_mask = np.load(sam_mask_path)

color = [[30 / 255, 144 / 255, 255 / 255, 0.6], [1.0, 0.749, 0/412, 0.6]]
for idx, image_path in enumerate(tqdm(image_paths)):
    # P,
    # if idx == 39:
    smpl_mask_idx = smpl_mask[idx]
    sam_mask_idx = sam_mask[idx]
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for p in range(sam_mask_idx.shape[0]):
        if p==0:
            show_mask(sam_mask_idx[p] > -.5, plt.gca(), color=color[p])
        else:
            show_mask(sam_mask_idx[p] > -.5, plt.gca(), color=color[p])
    # show_box(bounding_box, plt.gca())
    # show_points(input_point, input_label, plt.gca(), marker_size=30)
    plt.axis('off')
    plt.savefig(os.path.join(save_path_sam, '%04d.png' % idx), bbox_inches='tight', pad_inches=0.0)
    plt.close()

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    for p in range(smpl_mask_idx.shape[0]):
        show_mask(smpl_mask_idx[p] > 0.7, plt.gca(), color=color[p])
    # show_box(bounding_box, plt.gca())
    # show_points(input_point, input_label, plt.gca(), marker_size=30)
    plt.axis('off')
    plt.savefig(os.path.join(save_path_smpl, '%04d.png' % idx), bbox_inches='tight', pad_inches=0.0)
    plt.close()



        # soft_mask_path = soft_mask_path[idx]
        # soft_mask = cv2.imread(soft_mask_path)
        # # to gray
        # soft_mask = cv2.cvtColor(soft_mask, cv2.COLOR_BGR2GRAY)
        # soft_mask = soft_mask / 255.0
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image)
        # show_mask(soft_mask > 0.02, plt.gca(), color=color[0])
        # plt.axis('off')
        # plt.savefig(os.path.join(save_path_soft, '%04d_02.png' % idx), bbox_inches='tight', pad_inches=0.0)
        # plt.close()




# mask_paths = sorted(glob.glob(os.path.join(result_dir, 'test_mask/-1/*.png')))
# # normal_paths = sorted(glob.glob(os.path.join(result_dir, 'test_normal/-1/*.png')))
#
# start_frame = 0
# # end_frame = len(image_paths)
# end_frame = 85
#
# save_dir = os.path.join(result_dir, 'test_overlay_normal')
#
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
#
# for idx, image_path in enumerate(tqdm(image_paths[start_frame:end_frame])):
#     mask_path = mask_paths[idx]
#     normal_path = normal_paths[idx]
#
#     image = cv2.imread(image_path)
#     mask = (cv2.imread(mask_path)[:,:,-1] > 127)[:, :, np.newaxis]
#     normal = cv2.imread(normal_path)
#
#     overlay = (normal * mask + image * (1 - mask)).astype(np.uint8)
#     cv2.imwrite(os.path.join(save_dir, f'{idx:04d}.png'), overlay)
