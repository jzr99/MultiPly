import cv2
import numpy as np
import glob
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score, classification_report

def get_mask_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    iou = np.sum(intersection) / np.sum(union)
    return iou
def crop_image(img, bbox, batch=False):
    if batch:
        return img[:, int(bbox[1]):(int(bbox[1]) + int(bbox[3] - bbox[1])), int(bbox[0]): (int(bbox[0]) + int(bbox[2] - bbox[0]))]
    else:
        return img[int(bbox[1]):(int(bbox[1]) + int(bbox[3] - bbox[1])), int(bbox[0]): (int(bbox[0]) + int(bbox[2] - bbox[0]))]

gt_mask_dir = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/ground_truth'
gt_mask_paths = sorted(glob.glob(f'{gt_mask_dir}/*.png'))[:131]
our_mask_dir = '/home/chen/RGB-PINA/code/outputs/ThreeDPW/Helge_outdoor_wo_disp_freeze_20_every_20_opt_pose/test_mask'
RVM_mask_dir = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/RVM_masks'
SMPL_mask_dir = '/home/chen/RGB-PINA/data/Helge_outdoor/mask'
pointrend_mask_dir = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/pointrend'

resize_factor = 2
ours_precision = []
ours_recall = []
ours_f1 = []
ours_iou = []

RVM_precision = []
RVM_recall = []
RVM_f1 = []
RVM_iou = []

SMPL_precision = []
SMPL_recall = []
SMPL_f1 = []
SMPL_iou = []

pointrend_precision = []
pointrend_recall = []
pointrend_f1 = []
pointrend_iou = []

for gt_mask in tqdm(gt_mask_paths):
    frame_num = int(os.path.basename(gt_mask).split('.')[0][7:])
    gt_mask = cv2.imread(gt_mask, -1)[..., -1]
    # gt_mask = cv2.cvtColor(gt_mask, cv2.COLOR_GRAY2BGR)
    gt_mask = (cv2.resize(gt_mask, (gt_mask.shape[1] // resize_factor, gt_mask.shape[0] // resize_factor)) >= 255) 
    W = gt_mask.shape[1]
    H = gt_mask.shape[0]
    where = np.asarray(np.where(gt_mask))
    bbox_min = where.min(axis=1)
    bbox_min = bbox_min - 1
    bbox_max = where.max(axis=1)
    bbox_max = bbox_max + 1
    left, top, right, bottom = bbox_min[1], bbox_min[0], bbox_max[1], bbox_max[0]
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, W)
    bottom = min(bottom, H)
    crop_bbox = (left, top, right, bottom)

    gt_mask = crop_image(gt_mask, crop_bbox).reshape(-1).astype(np.uint8)
    ours_mask = crop_image((cv2.imread(f'{our_mask_dir}/{frame_num-300:04d}.png')[..., -1] == 255), crop_bbox).reshape(-1).astype(np.uint8)
    RVM_mask = crop_image((cv2.imread(f'{RVM_mask_dir}/{frame_num-300:04d}.png')[..., -1] == 255), crop_bbox).reshape(-1).astype(np.uint8)
    SMPL_mask = crop_image((cv2.imread(f'{SMPL_mask_dir}/{frame_num-300:04d}.png')[..., -1] == 255), crop_bbox).reshape(-1).astype(np.uint8)
    pointrend_mask = crop_image((cv2.imread(f'{pointrend_mask_dir}/{frame_num-300:04d}.png')[..., -1] == 255), crop_bbox).reshape(-1).astype(np.uint8)

    # print(classification_report(gt_mask, ours_mask, digits=5))
    # print(classification_report(gt_mask, RVM_mask, digits=5))

    # print(precision_score(gt_mask, ours_mask, average='macro'))
    # print(precision_score(gt_mask, RVM_mask, average='macro'))
    # print(recall_score(gt_mask, ours_mask, average='macro'))
    # print(recall_score(gt_mask, RVM_mask, average='macro'))
    # print(roc_auc_score(gt_mask, ours_mask))
    # print(roc_auc_score(gt_mask, RVM_mask))

    ours_precision.append(precision_score(gt_mask, ours_mask, average='macro'))
    # ours_recall.append(recall_score(gt_mask, ours_mask, average='macro'))
    ours_f1.append(f1_score(gt_mask, ours_mask, average='macro'))
    ours_iou.append(get_mask_iou(gt_mask, ours_mask))

    RVM_precision.append(precision_score(gt_mask, RVM_mask, average='macro'))
    # RVM_recall.append(recall_score(gt_mask, RVM_mask, average='macro'))
    RVM_f1.append(f1_score(gt_mask, RVM_mask, average='macro'))
    RVM_iou.append(get_mask_iou(gt_mask, RVM_mask))

    SMPL_precision.append(precision_score(gt_mask, SMPL_mask, average='macro'))
    # SMPL_recall.append(recall_score(gt_mask, SMPL_mask, average='macro'))
    SMPL_f1.append(f1_score(gt_mask, SMPL_mask, average='macro'))
    SMPL_iou.append(get_mask_iou(gt_mask, SMPL_mask))

    pointrend_precision.append(precision_score(gt_mask, pointrend_mask, average='macro'))
    # pointrend_recall.append(recall_score(gt_mask, pointrend_mask, average='macro'))
    pointrend_f1.append(f1_score(gt_mask, pointrend_mask, average='macro'))
    pointrend_iou.append(get_mask_iou(gt_mask, pointrend_mask))

print('ours_precision:', np.mean(ours_precision))
# print('ours_recall:', np.mean(ours_recall))
print('ours_f1:', np.mean(ours_f1))
print('ours_iou:', np.mean(ours_iou))

print('RVM_precision:', np.mean(RVM_precision))
# print('RVM_recall:', np.mean(RVM_recall))
print('RVM_f1:', np.mean(RVM_f1))
print('RVM_iou:', np.mean(RVM_iou))

print('SMPL_precision:', np.mean(SMPL_precision))
# print('SMPL_recall:', np.mean(SMPL_recall))
print('SMPL_f1:', np.mean(SMPL_f1))
print('SMPL_iou:', np.mean(SMPL_iou))

print('pointrend_precision:', np.mean(pointrend_precision))
# print('pointrend_recall:', np.mean(pointrend_recall))
print('pointrend_f1:', np.mean(pointrend_f1))
print('pointrend_iou:', np.mean(pointrend_iou))




import ipdb
ipdb.set_trace()
np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/ours_precision.npy', np.array(ours_precision))
np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/ours_f1.npy', np.array(ours_f1))
np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/ours_iou.npy', np.array(ours_iou))

np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/RVM_precision.npy', np.array(RVM_precision))
np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/RVM_f1.npy', np.array(RVM_f1))
np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/RVM_iou.npy', np.array(RVM_iou))

np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/SMPL_precision.npy', np.array(SMPL_precision))
np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/SMPL_f1.npy', np.array(SMPL_f1))
np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/SMPL_iou.npy', np.array(SMPL_iou))

np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/pointrend_precision.npy', np.array(pointrend_precision))
np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/pointrend_f1.npy', np.array(pointrend_f1))
np.save('/home/chen/RGB-PINA/data/Helge_outdoor/mask_evaluation/pointrend_iou.npy', np.array(pointrend_iou))



