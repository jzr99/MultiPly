import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

# subject = 'ma_wechat1'
# seq = f'{subject}'
# result_dir = f'/home/chen/release_tmp/RGB-PINA/outputs/Video/{seq}'
result_dir = '/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/dance5_sam_delay_depth_loop_noshare'
data_dir = f'/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/dance5'


image_paths = sorted(glob.glob(os.path.join(data_dir, 'image/*.png')))
mask_paths = sorted(glob.glob(os.path.join(result_dir, 'test_mask/-1/*.png')))
normal_paths = sorted(glob.glob(os.path.join(result_dir, 'test_normal/-1/*.png')))

start_frame = 0
end_frame = len(image_paths)
# end_frame = 85

save_dir = os.path.join(result_dir, 'test_overlay_normal_100')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for idx, image_path in enumerate(tqdm(image_paths[start_frame:end_frame])):
    mask_path = mask_paths[idx]
    normal_path = normal_paths[idx]

    image = cv2.imread(image_path)
    mask = (cv2.imread(mask_path)[:,:,-1] > 127)[:, :, np.newaxis]
    normal = cv2.imread(normal_path)

    overlay = (normal * mask + image * (1 - mask)).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'{idx:04d}.png'), overlay)