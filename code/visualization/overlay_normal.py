import cv2
import numpy as np
import glob
import os

subject = 'Weipeng_outdoor'
seq = f'{subject}_wo_disp_freeze_20_every_20'
result_dir = f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}'
data_dir = f'/home/chen/RGB-PINA/data/{subject}'


image_paths = sorted(glob.glob(os.path.join(data_dir, 'image/*.png')))
mask_paths = sorted(glob.glob(os.path.join(result_dir, 'test_mask/*.png')))
normal_paths = sorted(glob.glob(os.path.join(result_dir, 'test_normal/*.png')))

start_frame = 0
end_frame = len(image_paths)

save_dir = os.path.join(result_dir, 'test_overlay_normal')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for idx, image_path in enumerate(image_paths[start_frame:end_frame]):
    mask_path = mask_paths[idx]
    normal_path = normal_paths[idx]

    image = cv2.imread(image_path)
    mask = (cv2.imread(mask_path)[:,:,-1] > 127)[:, :, np.newaxis]
    normal = cv2.imread(normal_path)

    overlay = (normal * mask + image * (1 - mask)).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'{idx:04d}.png'), overlay)


