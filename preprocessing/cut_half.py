import cv2
import argparse
import glob


def main(args):
    seq = args.seq
    # gender = args.gender
    DIR = './raw_data'
    img_dir = f'{DIR}/{seq}/{seq}'   

    if args.source == 'hi4d':
        img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
    else:
        img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
    if len(img_paths) == 0:
        img_paths = sorted(glob.glob(f"{img_dir}/*.jpg"))
    for path in img_paths:
        img = cv2.imread(path)
        img = img[:, (img.shape[1] // 2):, :]
        # img = img[(img.shape[0] // 2)-100:-100, :, :]
        cv2.imwrite(path, img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocessing data")
    # video source
    parser.add_argument('--source', type=str, default='custom', help="custom video or dataset video")
    # sequence name
    parser.add_argument('--seq', type=str)
    args = parser.parse_args()
    main(args)