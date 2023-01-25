import cv2
import numpy as np
import os
from tqdm import trange
import glob
import imageio

seq = 'exstrimalik'
seq_dir = '/home/chen/RGB-PINA/code/outputs/ThreeDPW/exstrimalik_wo_disp_freeze_20_every_20_opt_pose'
ep1_frame = 0
ep2_frame = 198
# ep3_frame = 198
# ep4_frame = 274
rotate_frame = 198
image_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/data/exstrimalik', 'image/*.png')))
mask_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/exstrimalik_wo_disp_freeze_20_every_20_opt_pose', 'test_mask/*.png')))
normal_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/exstrimalik_wo_disp_freeze_20_every_20_opt_pose', 'test_normal/*.png')))
overlay_normal_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/exstrimalik_wo_disp_freeze_20_every_20_opt_pose', 'test_overlay_normal/*.png')))
fvr_mask_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/exstrimalik_wo_disp_freeze_20_every_20_opt_pose', 'test_fvr_mask_0198/*.png')))
fvr_normal_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/code/outputs/ThreeDPW/exstrimalik_wo_disp_freeze_20_every_20_opt_pose', 'test_fvr_normal_0198/*.png')))

image_writer = imageio.get_writer(os.path.join(seq_dir, 'ep1_1.mp4'), fps=30, macro_block_size=1)
overlay_normal_writer = imageio.get_writer(os.path.join(seq_dir, 'ep1_2.mp4'), fps=30, macro_block_size=1)
rotate_normal_writer = imageio.get_writer(os.path.join(seq_dir, 'ep1_3.mp4'), fps=30, macro_block_size=1)

for idx in range(len(image_paths)):
    if idx >= ep1_frame and idx <= ep2_frame:
        image_path = image_paths[idx]
        image_writer.append_data(imageio.imread(image_path))

        overlay_normal_path = overlay_normal_paths[idx]
        overlay_normal_writer.append_data(imageio.imread(overlay_normal_path))
    if idx == ep2_frame:
        overlay_normal_path = overlay_normal_paths[idx]
        overlay_normal = cv2.imread(overlay_normal_path)
        mask_path = mask_paths[idx]
        mask = (cv2.imread(mask_path)[:,:,-1] > 0)[:, :, np.newaxis]
        transparent_image = (overlay_normal * mask) + (np.ones_like(overlay_normal) * 255 * (1-mask)).astype(np.uint8)
        cv2.imwrite(os.path.join(f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/exstrimalik_wo_disp_freeze_20_every_20_opt_pose/{idx:04d}.png'), transparent_image)

    # if idx >= ep2_frame and idx < ep3_frame:
    #     overlay_normal_path = overlay_normal_paths[idx]
    #     overlay_normal_writer.append_data(imageio.imread(overlay_normal_path))
    # if idx == ep3_frame:
    #     for i in range(60):
    #         fvr_mask_path = fvr_mask_paths[i]
    #         fvr_normal_path = fvr_normal_paths[i]
    #         fvr_mask = (imageio.imread(fvr_mask_path) > 127)[:, :, np.newaxis]
    #         fvr_normal = imageio.imread(fvr_normal_path)
    #         rotate_normal = (fvr_normal * fvr_mask) + (np.ones_like(fvr_normal) * 255 * (1-fvr_mask)).astype(np.uint8)
    #         rotate_normal_writer.append_data(rotate_normal)


image_writer.close()
overlay_normal_writer.close()
# rotate_normal_writer.close()