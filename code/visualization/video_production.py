import cv2
import os
import imageio
# import glob
name = 'video5'
start_frame = 446
end_frame = 556

writer = imageio.get_writer('/home/chen/SelfReconCode/data/Invisible/masks/%s.mp4' % name, fps=30)

image_dir = '/home/chen/SelfReconCode/data/Invisible/masks'

files = [f for f in os.listdir(image_dir) if '.png' in f]
files.sort()

for f in files[start_frame:end_frame+1]:
    im = os.path.join(image_dir, f)
    writer.append_data(imageio.imread(im))
writer.close()

writer = imageio.get_writer('/home/chen/RGB-PINA/code/outputs/ThreeDPW/Invisible_off_weight_2_w_opt_smpl_normal_wo_detach/test_mask/%s.mp4' % name, fps=30)

image_dir = '/home/chen/RGB-PINA/code/outputs/ThreeDPW/Invisible_off_weight_2_w_opt_smpl_normal_wo_detach/test_mask'

files = [f for f in os.listdir(image_dir) if '.png' in f]
files.sort()

for f in files[start_frame:end_frame]:
    im = os.path.join(image_dir, f)
    writer.append_data(imageio.imread(im))
writer.close()