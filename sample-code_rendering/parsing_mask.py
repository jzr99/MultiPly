import numpy as np
import cv2
import os
import glob
from tqdm import tqdm
classes = {'background':(0,0,0), 'hat':(128,0,0), 'hair':(255,0,0), 'glove':(0,85,0), 'sunglasses':(170,0,51), 'upperclothes':(255,85,0),
           'dress':(0,0,85), 'coat':(0,119,221), 'socks':(85,85,0), 'pants':(0,85,85), 'jumpsuits':(85,51,0), 'scarf':(52,86,128), 'skirt':(0,128,0),
           'face':(0,0,255), 'leftArm':(51,170,221), 'rightArm':(0,255,255), 'leftLeg':(85,255,170), 'rightLeg':(170,255,85), 'leftShoe':(255,255,0),
           'rightShoe':(255,170,0)}

# NOte jumpsuits tend to the neck!!!
data_name = 'pss_resized'
seg_img_paths = [f for f in sorted(glob.glob(f'/home/chen/snarf_idr_cg_1/data/{data_name}/cloth_seg/*.png')) if 'gray' not in f]
save_dir = f'/home/chen/snarf_idr_cg_1/data/{data_name}/body_parsing'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
import ipdb; ipdb.set_trace()
for seg_img_path in tqdm(seg_img_paths):

    seg_img = cv2.imread(seg_img_path)
    blueC = seg_img[:,:,0]
    greenC = seg_img[:,:,1]
    redC = seg_img[:,:,2]
    parsing_mask = np.zeros_like(seg_img[..., 0])
    for parsing_class in ['face', 'leftArm', 'rightArm', 'leftLeg', 'rightLeg', 'jumpsuits']:
        class_color = classes[parsing_class][::-1] # RGB -> BGR
        mask = ((blueC == class_color[0]) & (greenC == class_color[1]) & (redC == class_color[2])) 
        parsing_mask[np.nonzero(mask)[0], np.nonzero(mask)[1]] = 1
    for parsing_class in ['hair']:
        class_color = classes[parsing_class][::-1] # RGB -> BGR
        mask = ((blueC == class_color[0]) & (greenC == class_color[1]) & (redC == class_color[2])) 
        parsing_mask[np.nonzero(mask)[0], np.nonzero(mask)[1]] = 2
    for parsing_class in ['upperclothes']:
        class_color = classes[parsing_class][::-1] # RGB -> BGR
        mask = ((blueC == class_color[0]) & (greenC == class_color[1]) & (redC == class_color[2])) 
        parsing_mask[np.nonzero(mask)[0], np.nonzero(mask)[1]] = 3
    for parsing_class in ['pants']:
        class_color = classes[parsing_class][::-1] # RGB -> BGR
        mask = ((blueC == class_color[0]) & (greenC == class_color[1]) & (redC == class_color[2])) 
        parsing_mask[np.nonzero(mask)[0], np.nonzero(mask)[1]] = 4
    for parsing_class in ['leftShoe', 'rightShoe']:
        class_color = classes[parsing_class][::-1] # RGB -> BGR
        mask = ((blueC == class_color[0]) & (greenC == class_color[1]) & (redC == class_color[2])) 
        parsing_mask[np.nonzero(mask)[0], np.nonzero(mask)[1]] = 5
    for parsing_class in ['background']:
        class_color = classes[parsing_class][::-1] # RGB -> BGR
        mask = ((blueC == class_color[0]) & (greenC == class_color[1]) & (redC == class_color[2])) 
        parsing_mask[np.nonzero(mask)[0], np.nonzero(mask)[1]] = 6
    cv2.imwrite(os.path.join(save_dir, seg_img_path.split('/')[-1]), parsing_mask)