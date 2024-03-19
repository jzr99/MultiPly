import numpy as np
import glob
import cv2
from sklearn.metrics import precision_score, roc_auc_score, recall_score, f1_score, classification_report
import os

def calculate_iou(mask_1, pred_mask):
    intersection = np.logical_and(mask_1, pred_mask)
    union = np.logical_or(mask_1, pred_mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# EXP_NAME = "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt"
# # EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose"
# # EXP_NAME = "Hi4D_pair19_piggyback19_temporal"
# # EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto"
# GT_NAME = "pair19/piggyback19"
# CAM_NAME = "4"
# inverse = True
EXP_NAME = "pair19_piggyback19_vitpose_4"
# EXP_NAME = "Hi4D_pair17_dance17_28_sam_delay_depth_loop_0pose_vitpose_2_noshare"
# EXP_NAME = "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_2_noshare"
# EXP_NAME = "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2_edge_noshare"
# EXP_NAME_test = "Hi4D_pair19_piggyback19_temporal"
# EXP_NAME = "Hi4D_pair19_piggyback19_temporal"
# EXP_NAME = "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto
# GT_NAME = "pair19/piggyback19"
GT_NAME = "pair19/piggyback19"
CAM_NAME = "4"
# inverse = False
# EXP_ROOT =  f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/{EXP_NAME}/stage_sam_mask/"
EXP_DIR = f"/media/ubuntu/hdd/easymocap/EasyMocap/3rdparty/Self-Correction-Human-Parsing/data/{EXP_NAME}/mask-schp/4"
GT_DIR = f"/media/ubuntu/hdd/Hi4D/{GT_NAME}/seg/img_seg_mask/{CAM_NAME}"

meta = dict(np.load(os.path.join(GT_DIR+"/../../..", "meta.npz")))
# self.num_persons = meta["num_persons"]
start = int(meta["start"])
end = int(meta["end"])
contact_frames = meta["contact_ids"].tolist()
# contact_frames = np.array(contact_frames) - start
print(len(contact_frames))
# import pdb;pdb.set_trace()
# sam_mask_list = sorted(glob.glob(EXP_DIR))
# sam_mask_test_list = [
#     EXP_ROOT + '00000',
#     EXP_ROOT + '01000',
#     EXP_ROOT + '02000',
# ]
# for sam_mask_path_i in sam_mask_test_list:
    # sam_mask_i = np.load(sam_mask_path_i + "/sam_opt_mask.npy")
iou_two_person_list = []
f1_two_person_list = []
precision_two_person_list = []
recall_two_person_list = []
# for person_i in range(2):
person_0_gt_path = GT_DIR + f"/0"
gt_mask_list_0 = sorted(glob.glob(person_0_gt_path + "/*.png"))
person_1_gt_path = GT_DIR + f"/1"
gt_mask_list_1 = sorted(glob.glob(person_1_gt_path + "/*.png"))
# assert sam_mask_i.shape[0] == len(gt_mask_list)
iou_score_list = []
f1_score_list = []
precision_list = []
recall_list = []
for idx, (gt_mask_path_i_0, gt_mask_path_i_1) in enumerate(zip(gt_mask_list_0, gt_mask_list_1)):
    if idx+start not in contact_frames:
        continue
    pred_mask_path_0 = EXP_DIR + f"/{idx:06d}_0.png"
    pred_mask_path_1 = EXP_DIR + f"/{idx:06d}_1.png"
    pred_mask_0 = cv2.imread(pred_mask_path_0)
    pred_mask_0 = cv2.cvtColor(pred_mask_0, cv2.COLOR_BGR2GRAY)
    pred_mask_1 = cv2.imread(pred_mask_path_1)
    pred_mask_1 = cv2.cvtColor(pred_mask_1, cv2.COLOR_BGR2GRAY)
    # import pdb;pdb.set_trace()
    mask_1 = cv2.imread(gt_mask_path_i_1)
    # preprocess: BGR -> Gray -> Mask -> Tensor
    mask_1 = cv2.cvtColor(mask_1, cv2.COLOR_BGR2GRAY)
    mask_1 = cv2.resize(mask_1, (pred_mask_1.shape[1], pred_mask_1.shape[0]))
    mask_1 = mask_1 > 0
    pred_mask_1 = pred_mask_1 > 0.0
    assert mask_1.shape == pred_mask_1.shape

    mask_0 = cv2.imread(gt_mask_path_i_0)
    # preprocess: BGR -> Gray -> Mask -> Tensor
    mask_0 = cv2.cvtColor(mask_0, cv2.COLOR_BGR2GRAY)
    mask_0 = cv2.resize(mask_0, (pred_mask_0.shape[1], pred_mask_0.shape[0]))
    mask_0 = mask_0 > 0
    pred_mask_0 = pred_mask_0 > 0.0
    assert mask_0.shape == pred_mask_0.shape
    # calculate iou
    iou_score_1 = calculate_iou(mask_1, pred_mask_1)
    iou_score_0 = calculate_iou(mask_0, pred_mask_0)
    iou_score_0011 = (iou_score_1 + iou_score_0)/2

    iou_score_1 = calculate_iou(mask_0, pred_mask_1)
    iou_score_0 = calculate_iou(mask_1, pred_mask_0)
    iou_score_0110 = (iou_score_1 + iou_score_0)/2

    os.makedirs(EXP_DIR + f"/../reorder/0", exist_ok=True)
    os.makedirs(EXP_DIR + f"/../reorder/1", exist_ok=True)
    if iou_score_0011 > iou_score_0110:
        mask_gt_1 = mask_1
        mask_gt_0 = mask_0
        iou_score = iou_score_0011
        cv2.imwrite(EXP_DIR + f"/../reorder/0/{idx:06d}.png", pred_mask_0*255)
        cv2.imwrite(EXP_DIR + f"/../reorder/1/{idx:06d}.png", pred_mask_1*255)
    else:
        mask_gt_1 = mask_0
        mask_gt_0 = mask_1
        iou_score = iou_score_0110
        cv2.imwrite(EXP_DIR + f"/../reorder/0/{idx:06d}.png", pred_mask_1*255)
        cv2.imwrite(EXP_DIR + f"/../reorder/1/{idx:06d}.png", pred_mask_0*255)

    # print(f"Epoch {epoch}, Person {person_i}, Frame {idx}, IoU {iou_score}")
    iou_score_list.append(iou_score)
    # import pdb;pdb.set_trace()
    f1_score_list.append(f1_score(mask_gt_1.reshape(-1), pred_mask_1.reshape(-1), average='macro'))
    precision_list.append(precision_score(mask_gt_1.reshape(-1), pred_mask_1.reshape(-1), average='macro'))
    recall_list.append(recall_score(mask_gt_1.reshape(-1), pred_mask_1.reshape(-1), average='macro'))
    f1_score_list.append(f1_score(mask_gt_0.reshape(-1), pred_mask_0.reshape(-1), average='macro'))
    precision_list.append(precision_score(mask_gt_0.reshape(-1), pred_mask_0.reshape(-1), average='macro'))
    recall_list.append(recall_score(mask_gt_0.reshape(-1), pred_mask_0.reshape(-1), average='macro'))

iou_two_person_list.append(np.mean(iou_score_list))
f1_two_person_list.append(np.mean(f1_score_list))
precision_two_person_list.append(np.mean(precision_list))
recall_two_person_list.append(np.mean(recall_list))
# epoch = sam_mask_path_i.split("/")[-1]
# print(f"EXP {EXP_NAME} Person {person_i}, Mean IoU {np.mean(iou_score_list)}")
print(f"two person mean iou: {np.mean(iou_two_person_list)}, f1: {np.mean(f1_two_person_list)}, precision: {np.mean(precision_two_person_list)}, recall: {np.mean(recall_two_person_list)}")