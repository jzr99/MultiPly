import sys
import cv2
import numpy as np
import torch.nn as nn
import detectron2
import pickle as pkl    
import glob
import os
from tqdm import tqdm

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
sys.path.insert(1, "/home/chen/detectron2/projects/PointRend")
import point_rend

class detectron2_seg(nn.Module):
    def __init__(self, seg_threshold=0.9):

        super().__init__()
        # create a detectron2 config and a detectron2 DefaultPredictor to run inference on this image
        # cfg_seg = get_cfg()
        # # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        # cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        # cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = seg_threshold  # set threshold for segmentation model
        # # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        # cfg_seg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        # self.seg_predictor = DefaultPredictor(cfg_seg)


        cfg_seg = get_cfg()
        # Add PointRend-specific config
        point_rend.add_pointrend_config(cfg_seg)
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        cfg_seg.merge_from_file("/home/chen/detectron2/projects/PointRend/configs/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco.yaml")
        cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = seg_threshold  # set threshold for segmentation model
        # Use a model from PointRend model zoo: https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend#pretrained-models
        cfg_seg.MODEL.WEIGHTS = "detectron2://PointRend/InstanceSegmentation/pointrend_rcnn_R_50_FPN_3x_coco/164955410/model_final_3c3198.pkl"
        self.seg_predictor = DefaultPredictor(cfg_seg)

    def forward(self, input_img):
        outputs = self.seg_predictor(input_img[:,:,:3])
        return outputs["instances"]

if __name__ == "__main__":
    d2_seg = detectron2_seg().cuda()
    source_dir = '/home/chen/RGB-PINA/data/Helge_outdoor/image'
    output_dir = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/pointrend'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    file_list = sorted(glob.glob(os.path.join(source_dir, '*.png')))
    for file in tqdm(file_list):
        img = cv2.imread(file)
        d2_seg_outputs = d2_seg(img)
        mask = d2_seg_outputs.pred_masks[0].float().cpu().numpy().squeeze()
        cv2.imwrite(os.path.join(output_dir, os.path.basename(file)), mask * 255)