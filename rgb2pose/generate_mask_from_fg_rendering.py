import os

import numpy as np
import glob
import cv2

# EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose_naive"
EXP_NAME_test = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose_naive"
# EXP_NAME_test = "Hi4D_pair19_piggyback19_temporal"
# EXP_NAME = "Hi4D_pair19_piggyback19_temporal"
# EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto
GT_NAME = "pair19/piggyback19"
CAM_NAME = "4"
inverse = False
# EXP_DIR = f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/{EXP_NAME}/stage_sam_mask/*"
EXP_TEST_DIR = f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/{EXP_NAME_test}/test_instance_mask"
GT_DIR = f"/cluster/project/infk/hilliges/jiangze/ROMP/global_scratch/romp_data/ROMP_datasets/Hi4D/Hi4D_all/Hi4D/{GT_NAME}/seg/img_seg_mask/{CAM_NAME}"
GT_IMG_DIR = f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/cycle_dance/image"
gt_img_list = sorted(glob.glob(GT_IMG_DIR + "/*.png"))
EXP_TEST_DIR = f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/cycle_dance_sam_ratio75/test_fg_rendering/"
# sam_mask_list = sorted(glob.glob(EXP_DIR))
# for epoch, sam_mask_path_i in enumerate(sam_mask_list):
# sam_mask_i = np.load(sam_mask_path_i + "/sam_opt_mask.npy")
iou_two_person_list = []
for person_i in range(2):
    if inverse:
        person_i_test_path = EXP_TEST_DIR + f"/{1-person_i}"
    else:
        person_i_test_path = EXP_TEST_DIR + f"/{person_i}"
    # person_i_gt_path = GT_DIR + f"/{person_i}"

    # gt_mask_list = sorted(glob.glob(person_i_gt_path + "/*.png"))
    test_mask_list = sorted(glob.glob(person_i_test_path + "/*.png"))
    # assert sam_mask_i.shape[0] == len(gt_img_list)
    iou_score_list = []
    for idx, gt_mask_path_i in enumerate(gt_img_list):
        # import pdb;pdb.set_trace()
        mask_1 = cv2.imread(gt_mask_path_i)
        # preprocess: BGR -> Gray -> Mask -> Tensor
        # mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)
        mask_test = cv2.imread(test_mask_list[idx])
        # mask_test = cv2.cvtColor(mask_test, cv2.COLOR_BGR2GRAY)
        # if inverse:
        #     pred_mask = sam_mask_i[idx, 1-person_i, :, :]
        # else:
        #     pred_mask = sam_mask_i[idx, person_i, :, :]
        error_mask = np.linalg.norm(mask_1[:,:,:3] - mask_test[:,:,:3], axis=2)
        excluded_mask = np.min(mask_test[:,:,:3], axis=2) < 250
        error_mask = error_mask < 80
        depth_mask = np.logical_and(error_mask, excluded_mask)
        os.makedirs(EXP_TEST_DIR+ f"/error_mask_{person_i}", exist_ok=True)
        cv2.imwrite(EXP_TEST_DIR+ f"/error_mask_{person_i}" + f"/{idx}.png", (depth_mask)*255)
        # resize gt mask to pred_mask
        # mask_1 = cv2.resize(mask_1, (pred_mask.shape[1], pred_mask.shape[0]))
        # mask_test = mask_test > 0
        # mask_1 = mask_1 > 0
        # pred_mask = pred_mask > 0.0
        # # assert mask_1.shape == pred_mask.shape
        # assert mask_test.shape == mask_1.shape
        # pred_mask = mask_test
        # # calculate iou
        # intersection = np.logical_and(mask_1, pred_mask)
        # union = np.logical_or(mask_1, pred_mask)
        # iou_score = np.sum(intersection) / np.sum(union)
        # # print(f"Epoch {epoch}, Person {person_i}, Frame {idx}, IoU {iou_score}")
        # iou_score_list.append(iou_score)
    # iou_two_person_list.append(np.mean(iou_score_list))
    # print(f"EXP {EXP_NAME} Epoch {epoch * 50}, Person {person_i}, Mean IoU {np.mean(iou_score_list)}")
    # print(f"two person mean iou: {np.mean(iou_two_person_list)}")

