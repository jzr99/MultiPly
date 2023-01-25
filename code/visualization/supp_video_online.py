import cv2
import numpy as np
import os
from tqdm import trange
import glob
import imageio

image_paths = sorted(glob.glob('/home/chen/RGB-PINA/data/Invisible/image/*.png'))
ours_paths = sorted(glob.glob('/home/chen/RGB-PINA/code/outputs/ThreeDPW/Invisible_wo_disp_freeze_20_every_20_opt_pose/test_overlay_normal/*.png'))
icon_paths = sorted(glob.glob('/home/chen/disk2/ICON_new_results/Invisible/icon-filter/recon_overlay/*.png'))
selfrecon_paths = sorted(glob.glob('/home/chen/disk2/SelfRecon_results/Invisible/result/meshs/*.png'))
seq_dir = '/home/chen/RGB-PINA/code/outputs/ThreeDPW/Invisible_wo_disp_freeze_20_every_20_opt_pose'
image_1_writer = imageio.get_writer(os.path.join(seq_dir, 'image_1.mp4'), fps=30, macro_block_size=1)
# image_2_writer = imageio.get_writer(os.path.join(seq_dir, 'image_2.mp4'), fps=30, macro_block_size=1)
ours_1_writer = imageio.get_writer(os.path.join(seq_dir, 'ours_1.mp4'), fps=30, macro_block_size=1)
# ours_2_writer = imageio.get_writer(os.path.join(seq_dir, 'ours_2.mp4'), fps=30, macro_block_size=1)
icon_1_writer = imageio.get_writer(os.path.join(seq_dir, 'icon_1.mp4'), fps=30, macro_block_size=1)
# icon_2_writer = imageio.get_writer(os.path.join(seq_dir, 'icon_2.mp4'), fps=30, macro_block_size=1)
selfrecon_1_writer = imageio.get_writer(os.path.join(seq_dir, 'selfrecon_1.mp4'), fps=30, macro_block_size=1)
# selfrecon_2_writer = imageio.get_writer(os.path.join(seq_dir, 'selfrecon_2.mp4'), fps=30, macro_block_size=1)

start_frame = 0
# milestone_frame = 682 + 1
end_frame = 346 + 1

for idx, image_path in enumerate(image_paths):

    if idx >= start_frame and idx < end_frame:
        ours_path = ours_paths[idx]
        icon_path = icon_paths[idx]
        selfrecon_path = selfrecon_paths[idx]
        
        image_1_writer.append_data(imageio.imread(image_path))
        ours_1_writer.append_data(imageio.imread(ours_path))
        icon_1_writer.append_data(imageio.imread(icon_path))
        selfrecon_1_writer.append_data(imageio.imread(selfrecon_path))

image_1_writer.close()
# image_2_writer.close()
ours_1_writer.close()
# ours_2_writer.close()
icon_1_writer.close()
# icon_2_writer.close()
selfrecon_1_writer.close()
# selfrecon_2_writer.close()
        
            