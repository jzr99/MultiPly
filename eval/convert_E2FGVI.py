import cv2
import numpy as np
import glob
import os

E2FGVI_raw_dir = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/E2FGVI_raw'
E2FGVI_dir = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/E2FGVI_bg'
E2FGVI_bg_paths = sorted(glob.glob(f'{E2FGVI_dir}/*.png'))
E2FGVI_raw_paths = sorted(glob.glob(f'{E2FGVI_raw_dir}/*.png'))
output_dir = '/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/E2FGVI_mask'

if not os.path.exists(output_dir):
    os.makedirs(output_dir) 

for idx, E2FGVI_bg_path in enumerate(E2FGVI_bg_paths):
    E2FGVI_bg = cv2.imread(E2FGVI_bg_path)
    E2FGVI_raw = cv2.imread(E2FGVI_raw_paths[idx])
    E2FGVI_diff = np.all((E2FGVI_raw - E2FGVI_bg) > 0, axis=2)
    import ipdb
    ipdb.set_trace()
    cv2.imwrite(os.path.join(output_dir, os.path.basename(E2FGVI_bg_path)), E2FGVI_diff * 255)