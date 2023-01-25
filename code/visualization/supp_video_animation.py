import cv2
import numpy as np
import os
from tqdm import trange
import glob
import imageio
['Helge_outdoor', 'Lan_2', 'Nadia_outdoor', 'messi_1', 'Invisible', 'parkinglot', 'exstrimalik', 'seattle', 'manuel_outdoor']
for seq in ['roger']: # 'parkinglot', 'seattle
    print(seq)
    if seq == 'Helge_outdoor':
        seq_dir = '/home/chen/RGB-PINA/code/outputs/ThreeDPW/Helge_outdoor_wo_disp_freeze_20_every_20_opt_pose_cano_mesh'
    elif seq == 'Invisible':
        seq_dir = '/home/chen/RGB-PINA/code/outputs/ThreeDPW/Invisible_wo_disp_freeze_20_every_20_opt_pose'
    elif seq == 'exstrimalik':
        seq_dir = '/home/chen/RGB-PINA/code/outputs/ThreeDPW/exstrimalik_wo_disp_freeze_20_every_20_opt_pose'
    elif seq == 'roger':
        seq_dir = '/home/chen/RGB-PINA/code/outputs/ThreeDPW/roger_wo_disp_freeze_20_every_20_opt_pose'
    else:
        seq_dir = f'/home/chen/disk2/RGB-PINA_results/{seq}_wo_disp_freeze_20_every_20_opt_pose'

    fg_rendering_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_animation_fg_rendering', '*.png')))
    normal_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_animation_normal', '*.png')))
    mask_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_animation_mask', '*.png')))
    fg_rendering_writer = imageio.get_writer(os.path.join(seq_dir, 'animation_fg_rendering.mp4'), fps=30, macro_block_size=1)
    normal_writer = imageio.get_writer(os.path.join(seq_dir, 'animation_normal.mp4'), fps=30, macro_block_size=1)

    for idx, normal_path in enumerate(normal_paths):

        fg_rendering_path = fg_rendering_paths[idx]
        mask_path = mask_paths[idx]
        mask = (imageio.imread(mask_path) > 100)[:, :, np.newaxis]
        normal = imageio.imread(normal_path)
        normal = ((normal * mask) + (np.ones_like(normal) * 255 * (1 - mask))).astype(np.uint8)
        
        fg_rendering_writer.append_data(imageio.imread(fg_rendering_path))
        normal_writer.append_data(normal)

    fg_rendering_writer.close()
    normal_writer.close()
