# Inference with a panoptic segmentation model
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random


# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
predictor = DefaultPredictor(cfg)
image_dir = '/home/chen/disk2/3DPW/imageFiles/outdoors_freestyle_00'
output_dir = '/home/chen/disk2/3DPW/ground_mask/outdoors_freestyle_00'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
import glob
from tqdm import tqdm
image_paths = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))
for image_path in tqdm(image_paths):
    im = cv2.imread(image_path)
    img_tmp = np.zeros_like(im)
    panoptic_seg, segments_info = predictor(im)["panoptic_seg"]
    segment_ids = []
    for s in segments_info:
        if s['category_id'] == 44 or s['category_id'] == 47 or s['category_id'] == 51 or s['category_id'] == 13 or s['category_id'] == 52:
            segment_ids.append(s['id'])
    for segment_id in segment_ids:
        groud_index = np.where(panoptic_seg.data.cpu().numpy() == segment_id)
        img_tmp[groud_index] = 255
    cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), img_tmp)

    # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
    # cv2.imwrite(os.path.join(output_dir, os.path.basename(image_path)), out.get_image()[:, :, ::-1])

# visualization of different parts
# import ipdb
# ipdb.set_trace()
# for i in range(int(panoptic_seg.max().cpu().numpy())+1):

#     im_tmp = im.copy()
#     index = np.where(panoptic_seg.data.cpu().numpy() == i)
#     im_tmp[index] = 0
#     cv2.imwrite(f'image_2_index_{i}.png', im_tmp)

