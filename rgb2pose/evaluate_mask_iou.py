import numpy as np
import glob
import cv2
import os
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score, classification_report
# EXP_NAME = "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt"
# # EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose"
# # EXP_NAME = "Hi4D_pair19_piggyback19_temporal"
# # EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto"
# GT_NAME = "pair19/piggyback19"
# CAM_NAME = "4"
# inverse = True

# EXP_NAME = "Hi4D_pair19_piggyback19_4_sam_delay_depth_2_noshare"
EXP_NAME = "Hi4D_pair17_dance17_28_sam_delay_depth_loop_0pose_vitpose_2_noshare"
# EXP_NAME = "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_2_noshare"
# EXP_NAME = "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2_edge_noshare"
# EXP_NAME_test = "Hi4D_pair19_piggyback19_temporal"
# EXP_NAME = "Hi4D_pair19_piggyback19_temporal"
# EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto
GT_NAME = "pair17/dance17"
CAM_NAME = "28"
inverse = False

EXP_ROOT =  f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/{EXP_NAME}/stage_sam_mask/"
EXP_DIR = f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/{EXP_NAME}/stage_sam_mask/*"
GT_DIR = f"/cluster/project/infk/hilliges/jiangze/ROMP/global_scratch/romp_data/ROMP_datasets/Hi4D/Hi4D_all/Hi4D/{GT_NAME}/seg/img_seg_mask/{CAM_NAME}"
meta = dict(np.load(os.path.join(GT_DIR+"/../../..", "meta.npz")))
start = int(meta["start"])
end = int(meta["end"])
contact_frames = meta["contact_ids"].tolist()
print("number of contact frames: ", len(contact_frames))
sam_mask_list = sorted(glob.glob(EXP_DIR))
sam_mask_test_list = [
    EXP_ROOT + '00000',
    EXP_ROOT + '01000',
    EXP_ROOT + '02000',
]
for sam_mask_path_i in sam_mask_test_list:
    sam_mask_i = np.load(sam_mask_path_i + "/sam_opt_mask.npy")
    iou_two_person_list = []
    f1_two_person_list = []
    precision_two_person_list = []
    recall_two_person_list = []
    for person_i in range(2):
        person_i_gt_path = GT_DIR + f"/{person_i}"
        gt_mask_list = sorted(glob.glob(person_i_gt_path + "/*.png"))
        assert sam_mask_i.shape[0] == len(gt_mask_list)
        iou_score_list = []
        f1_score_list = []
        precision_list = []
        recall_list = []
        for idx, gt_mask_path_i in enumerate(gt_mask_list):
            if idx + start not in contact_frames:
                continue
            # import pdb;pdb.set_trace()
            mask_1 = cv2.imread(gt_mask_path_i)
            # preprocess: BGR -> Gray -> Mask -> Tensor
            mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)
            if inverse:
                pred_mask = sam_mask_i[idx, 1-person_i, :, :]
            else:
                pred_mask = sam_mask_i[idx, person_i, :, :]
            # resize gt mask to pred_mask
            mask_1 = cv2.resize(mask_1, (pred_mask.shape[1], pred_mask.shape[0]))
            mask_1 = mask_1 > 0
            pred_mask = pred_mask > 0.0
            assert mask_1.shape == pred_mask.shape
            # calculate iou
            intersection = np.logical_and(mask_1, pred_mask)
            union = np.logical_or(mask_1, pred_mask)
            iou_score = np.sum(intersection) / np.sum(union)
            # print(f"Epoch {epoch}, Person {person_i}, Frame {idx}, IoU {iou_score}")
            iou_score_list.append(iou_score)
            # import pdb;pdb.set_trace()
            f1_score_list.append(f1_score(mask_1.reshape(-1), pred_mask.reshape(-1), average='macro'))
            precision_list.append(precision_score(mask_1.reshape(-1), pred_mask.reshape(-1), average='macro'))
            recall_list.append(recall_score(mask_1.reshape(-1), pred_mask.reshape(-1), average='macro'))

        iou_two_person_list.append(np.mean(iou_score_list))
        f1_two_person_list.append(np.mean(f1_score_list))
        precision_two_person_list.append(np.mean(precision_list))
        recall_two_person_list.append(np.mean(recall_list))
        epoch = sam_mask_path_i.split("/")[-1]
        print(f"EXP {EXP_NAME} Epoch {epoch}, Person {person_i}, Mean IoU {np.mean(iou_score_list)}")
    print(f"two person mean iou: {np.mean(iou_two_person_list)}, f1: {np.mean(f1_two_person_list)}, precision: {np.mean(precision_two_person_list)}, recall: {np.mean(recall_two_person_list)}")

