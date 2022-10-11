import cv2
import numpy as np
import glob
import os
import imageio


actor = 'Suarez'
seq_dir = f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{actor}_wo_disp_freeze_20_every_20_opt_pose'

image_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/data', actor, 'image/*.png')))
fg_rendering_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_fg_rendering/*.png'))) 
normal_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_normal/*.png')))
mask_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_mask/*.png')))

ep1_start_frame = 0
ep2_start_frame = 120
ep3_start_frame = 267
ep4_start_frame = 400

fake_bg_writer = imageio.get_writer(os.path.join(seq_dir, f'{actor}_fake_bg.mp4'), fps=30)
for idx in range(len(image_paths)):
    image_path = image_paths[idx]
    fg_rendering_path = fg_rendering_paths[idx]
    normal_path = normal_paths[idx]
    # (imageio.imread(mask_paths[idx]) == 255)

    if idx > ep4_start_frame:
        fg_rendering = cv2.imread(fg_rendering_path)
        fg_rendering = cv2.resize(fg_rendering, (fg_rendering.shape[1] // 4, fg_rendering.shape[0] // 4))
        mask = (cv2.imread(mask_paths[idx]) == 255)[:, :, -1].astype(np.uint8)

        mask = cv2.resize(mask, (mask.shape[1] // 4, mask.shape[0] // 4)).astype(np.bool)

        bg_image = cv2.imread('/home/chen/Downloads/Sky.jpg')
        bg_image = cv2.resize(bg_image, (960, 540))

        bg_image[(np.where(mask)[0] + 350, np.where(mask)[1] + 270)] = fg_rendering[mask]
        fake_bg_writer.append_data(bg_image[..., [2,1,0]])


fake_bg_writer.close()