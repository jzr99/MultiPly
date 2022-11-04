import cv2
import numpy as np
import glob
import os

def mask_out_ours():
    subject = 'Helge_outdoor'
    seq = f'{subject}_wo_disp_freeze_20_every_20_opt_pose'
    result_dir = f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}'
    data_dir = f'/home/chen/RGB-PINA/data/{subject}'


    image_paths = sorted(glob.glob(os.path.join(data_dir, 'image/*.png')))
    mask_paths = sorted(glob.glob(os.path.join(result_dir, 'test_mask/*.png')))
    normal_paths = sorted(glob.glob(os.path.join(result_dir, 'test_normal/*.png')))

    start_frame = 0
    end_frame = len(image_paths)

    save_dir = os.path.join(result_dir, 'RVM_mask_out')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for idx, image_path in enumerate(image_paths[start_frame:end_frame]):
        mask_path = mask_paths[idx]
        normal_path = normal_paths[idx]

        image = cv2.imread(image_path)
        mask = (cv2.imread(mask_path)[:,:,-1] == 255)[:, :, np.newaxis]
        # normal = cv2.imread(normal_path)

        overlay = (image * mask).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{idx:04d}.png'), overlay)

def mask_out_RVM():
    image_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/data/Helge_outdoor/image/*.png')))
    mask_paths = sorted(glob.glob('/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/RVM_masks/*.png'))

    save_dir = os.path.join('/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/RVM_mask_out')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for idx, image_path in enumerate(image_paths):
        mask_path = mask_paths[idx]

        image = cv2.imread(image_path)
        mask = (cv2.imread(mask_path)[:,:,-1] == 255)[:, :, np.newaxis]

        overlay = (image * mask).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{idx:04d}.png'), overlay)

def mask_out_ds():
    image_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/data/Helge_outdoor/image/*.png')))
    mask_paths = sorted(glob.glob('/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/sprites_mask/*.png'))

    save_dir = os.path.join('/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/sprites_mask_out')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for idx, image_path in enumerate(image_paths):
        mask_path = mask_paths[idx]

        image = cv2.imread(image_path)
        mask = (cv2.imread(mask_path)[:,:,-1] > 127)[:, :, np.newaxis]

        overlay = (image * mask).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{idx:04d}.png'), overlay)

def mask_out_pointrend():
    image_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/data/Helge_outdoor/image/*.png')))
    mask_paths = sorted(glob.glob('/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/pointrend/*.png'))

    save_dir = os.path.join('/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Helge_outdoor/pointrend_mask_out')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for idx, image_path in enumerate(image_paths):
        mask_path = mask_paths[idx]

        image = cv2.imread(image_path)
        mask = (cv2.imread(mask_path)[:,:,-1] > 255)[:, :, np.newaxis]

        overlay = (image * mask).astype(np.uint8)
        cv2.imwrite(os.path.join(save_dir, f'{idx:04d}.png'), overlay)

if __name__ == '__main__':
    # mask_out_ours()
    # mask_out_RVM()
    mask_out_ds()
    # mask_out_pointrend()