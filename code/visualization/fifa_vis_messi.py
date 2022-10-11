import cv2
import numpy as np
import glob
import os
import imageio


actor = 'messi_1'
seq_dir = f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{actor}_wo_disp_freeze_20_every_20_opt_pose'

image_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/data', actor, 'image/*.png')))
smpl_overlay_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/data', actor, 'joint_opt_smpl/*.png')))
fg_rendering_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_fg_rendering/*.png'))) 
normal_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_normal/*.png')))
mask_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_mask/*.png')))
fvr_rendering_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_fvr/*.png')))
fvr_normal_paths = sorted(glob.glob(os.path.join(seq_dir, 'test_fvr_normal/*.png')))

ep1_start_frame = 0
ep2_start_frame = 120
ep3_start_frame = 300
ep4_start_frame = 554
image_writer = imageio.get_writer(os.path.join(seq_dir, f'{actor}_ep1.mp4'), fps=30)
smpl_overlay_writer = imageio.get_writer(os.path.join(seq_dir, f'{actor}_ep2.mp4'), fps=30)
# fg_rendering_writer = imageio.get_writer(os.path.join(seq_dir, f'{actor}_ep2.mp4'), fps=30)
normal_writer = imageio.get_writer(os.path.join(seq_dir, f'{actor}_ep3.mp4'), fps=30)
# fake_bg_writer = imageio.get_writer(os.path.join(seq_dir, f'{actor}_ep4.mp4'), fps=30)
test_fvr_writer = imageio.get_writer(os.path.join(seq_dir, f'{actor}_fvr.mp4'), fps=15)
test_fvr_normal_writer = imageio.get_writer(os.path.join(seq_dir, f'{actor}_fvr_normal.mp4'), fps=15)

# for idx in range(len(image_paths)):
#     image_path = image_paths[idx]
#     smpl_overlay_path = smpl_overlay_paths[idx]
#     fg_rendering_path = fg_rendering_paths[idx]
#     normal_path = normal_paths[idx]
#     mask = (imageio.imread(mask_paths[idx]) == 255)

#     if idx > ep1_start_frame and idx < ep2_start_frame:
#         image_writer.append_data(imageio.imread(image_path))
#     elif idx > ep2_start_frame and idx < ep3_start_frame:
#         # fg_rendering_writer.append_data(imageio.imread(fg_rendering_path))
#         smpl_overlay_writer.append_data(imageio.imread(smpl_overlay_path))
#     elif idx > ep3_start_frame and idx < ep4_start_frame:
#         image = imageio.imread(image_path)
#         normal = imageio.imread(normal_path)
#         masked_normal = normal * mask[..., None] + image * (1 - mask[..., None]) 
#         normal_writer.append_data(masked_normal.astype(np.uint8))
#     # else:
#     #     fg_rendering = cv2.imread(fg_rendering_path)
#     #     bg_image = cv2.imread('/home/chen/Downloads/ranking200121.jpg')
#     #     bg_image = cv2.resize(bg_image, (960, 540))
#     #     bg_image[mask] = fg_rendering[mask]
#     #     fake_bg_writer.append_data(bg_image[..., [2,1,0]])

# image_writer.close()
# smpl_overlay_writer.close()
# # fg_rendering_writer.close()
# normal_writer.close()
# # fake_bg_writer.close()


for i in range(0, 60):
    test_fvr_writer.append_data(imageio.imread(fvr_rendering_paths[i]))
    test_fvr_normal_writer.append_data(imageio.imread(fvr_normal_paths[i]))

test_fvr_writer.close()
test_fvr_normal_writer.close()