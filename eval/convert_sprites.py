import cv2
import glob
import os

sprites_mask_raw_dir = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/sprites_mask_raw'
save_dir = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/sprites_mask'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
mask_paths = sorted(glob.glob(sprites_mask_raw_dir + '/*.png'))

for mask_path in mask_paths:
    mask = cv2.imread(mask_path)
    mask = cv2.resize(mask, (960, 560))
    cv2.imwrite(os.path.join(save_dir, os.path.basename(mask_path)), mask)