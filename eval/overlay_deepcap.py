import cv2
import numpy as np
import glob
import os
seq = 'Marc_1'
DIR = f'/home/chen/disk2/MPI_INF_Dataset/DeepCapDataset/DeepCapResults/{seq}'
output_dir = f'/home/chen/disk2/MPI_INF_Dataset/DeepCapDataset/{seq}/DeepCap_Overlay'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
start_frame = 2560
# end_frame = 1926

image_dir = f'/home/chen/RGB-PINA/data/{seq}/image'

image_paths = sorted(glob.glob(os.path.join(image_dir, '*.png')))
# rendered_paths = sorted(glob.glob(f'{DIR}/*.png'))[start_frame:start_frame + len(image_paths)]


for idx in range(len(image_paths)):
    rendered_path = f'/home/chen/disk2/MPI_INF_Dataset/DeepCapDataset/DeepCapResults/{seq}/image_c_0_f_{(idx + start_frame):04d}.png'
    rendered = cv2.imread(rendered_path, -1) # cv2.imread(rendered_path, -1)


    image = cv2.imread(image_paths[idx])

    rendered = cv2.resize(rendered, (image.shape[1], image.shape[0]))
    valid_mask = (rendered[:, :, 3] > 0)[:, :, np.newaxis]
    rendered_rgb = rendered[:, :, :3]
    overlay = (rendered_rgb * valid_mask + image * (1 - valid_mask)).astype(np.uint8)
    cv2.imwrite(os.path.join(output_dir, '%04d.png' % idx), overlay)