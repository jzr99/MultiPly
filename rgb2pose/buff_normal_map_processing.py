import cv2
import os

ref_size = 512 # default PIFuHD resolution
raw_size = (480, 640) 

pifuhd_output_path = '/home/chen/Semester-Project/perfcap_pifu_no_tex/results/pifuhd_final/recon/buff_gerard'
file_paths = [f for f in os.listdir(pifuhd_output_path) if 'png' in f]


for i in range(len(file_paths)):
    input_image = cv2.imread(f'/home/chen/Semester-Project/perfcap_pifu_no_tex/results/pifuhd_final/recon/buff_gerard/result_buff_gerard_308_256.png')
    color_image = input_image[:, :ref_size]
    front_normal_map = input_image[:, ref_size:ref_size*2]


    front_normal_map = cv2.resize(front_normal_map, raw_size)

    cv2.imwrite(f'/home/chen/disk2/DSFN_dataset/datasets/buff_rgbd/pred_normal/{308:04d}.png', front_normal_map)
    mask = cv2.imread(f'/home/chen/disk2/DSFN_dataset/datasets/buff_rgbd/mask/mask_{308:04d}.png')

    mask = (mask > 0)
    masked_normal_map = front_normal_map * mask[:, :, :1]
    cv2.imwrite(f'/home/chen/disk2/DSFN_dataset/datasets/buff_rgbd/pred_normal/masked_{308:04d}.png', masked_normal_map)

    # cv2.imwrite(f'/home/chen/snarf_idr_cg_1/data/pss_resized/normal/{i:04d}.png', front_normal_map)