import cv2
import argparse
import glob
import os
import numpy as np


def main(args):
    seq = args.seq
    # gender = args.gender
    # DIR = './raw_data'
    # img_dir = f'{DIR}/{seq}/{seq}'   
    img_dir = args.seq

    if args.source == 'hi4d':
        img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    elif args.source == 'econ':
        img_paths = sorted(glob.glob(f"{img_dir}/econ/png/*_cloth.png"))
        img_paths_overlap = sorted(glob.glob(f"{img_dir}/econ/png/*_overlap.png"))
        img_paths = img_paths[0:number_frames]
        img_paths_overlap = img_paths_overlap[0:number_frames]
    else:
        img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    # if len(img_paths) == 0:
    #     img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
    save_path = os.path.join(img_dir, 'crop')
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    for path in img_paths:

        file_name=os.path.basename(path)
        img = cv2.imread(path)
        white = np.array([127, 127, 127])
        # calculate the distance between the white color and the pixel color
        distance = np.linalg.norm(img - white, axis=-1)
        # set all pixels that are close enough to white to be white
        

        mask = np.all(img == white, axis=-1)
        # import pdb;pdb.set_trace()
        img[mask] = [255, 255, 255]
        # use np tp get bounding box from mask
        # import pdb;pdb.set_trace()
        coords = np.argwhere(~mask)

        img[distance < 10] = [255, 255, 255]
        center_x = int(np.mean(coords[:, 1]))
        center_y = int(np.mean(coords[:, 0]))
        # x_min, y_min = coords.min(axis=0)
        # x_max, y_max = coords.max(axis=0)
        # w = x_max - x_min + 1
        # h = y_max - y_min + 1
        # center_x = x_min + w // 2
        # center_y = y_min + h // 2
        # draw a square of size 128 around the center
        x1 = center_x - 128
        x2 = center_x + 128
        y1 = center_y - 230
        y2 = center_y + 230
        # x1 = center_x - 108
        # x2 = center_x + 108
        # y1 = center_y - 148
        # y2 = center_y + 148
        # crop the image based on the square
        img = img[y1:y2, x1:x2, :]
        # pad the image to make it square with write color
        # import pdb;pdb.set_trace()
        h, w, _ = img.shape
        if h > w:
            pad = (h - w) // 2
            pad_1 = h -w - pad
            img = np.pad(img, ((0, 0), (pad, pad_1), (0, 0)), 'constant', constant_values=255)
            # img = np.pad(img, ((0, 0), (pad, pad), (0, 0)), 'constant', constant_values=255)
        elif w > h:
            pad = (w - h) // 2
            pad_1 = w - h - pad
            img = np.pad(img, ((pad, pad_1), (0, 0), (0, 0)), 'constant', constant_values=255)
            # img = np.pad(img, ((pad, pad), (0, 0), (0, 0)), 'constant', constant_values=255)
        cv2.imwrite(save_path + f'/{file_name}', img)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing data")
    # video source
    parser.add_argument('--source', type=str, default='hi4d', help="custom video or dataset video")
    # sequence name
    parser.add_argument('--seq', type=str)
    args = parser.parse_args()

    # location = [3,3,4,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,4,4,4,4,4,4,4,4,3,3,2,3,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,4,4,4,4,3,3,3,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,3,4,4,3,4,3,4,4,4,4,4,3,3,3,3,3,4,4,4,4,4,4,4,4,4,] # dance4
    # location = [1 for _ in range(4)] + [2 for _ in range(2)] + [1 for _ in range(10)] + [2 for _ in range(10)]+[1,]+ [2 for _ in range(31)] + [1 for _ in range(2)] + [2 for _ in range(22)] + [1,1, 2,2,2,1,2,2] # jump16
    # location = [1 for _ in range(150)]  # alehug
    main(args)