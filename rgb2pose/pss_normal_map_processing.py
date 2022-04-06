import cv2
import os

ref_size = 512 # default PIFuHD resolution
raw_size = 540 # 1080 // 2

pifuhd_output_path = '/home/chen/Semester-Project/perfcap_pifu_no_tex/results/pifuhd_final/recon/pss_male-4-casual'
file_paths = [f for f in os.listdir(pifuhd_output_path) if 'png' in f]


for i in range(len(file_paths)):
    input_image = cv2.imread(f'/home/chen/Semester-Project/perfcap_pifu_no_tex/results/pifuhd_final/recon/pss_male-4-casual/result_pss_male-4-casual_{i}_256.png')
    color_image = input_image[:, :ref_size]
    front_normal_map = input_image[:, ref_size:ref_size*2]


    front_normal_map = cv2.resize(front_normal_map, (raw_size, raw_size))
    mask = cv2.imread(f'/home/chen/snarf_idr_cg_1/data/pss_resized/mask/{i:04d}.png')
    mask = (mask > 0)
    masked_normal_map = front_normal_map * mask[:, :, :1]

    cv2.imwrite(f'/home/chen/snarf_idr_cg_1/data/pss_resized/normal/{i:04d}.png', front_normal_map)