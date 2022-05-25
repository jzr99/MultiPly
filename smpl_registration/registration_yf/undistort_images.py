import argparse
import glob
import numpy as np
import cv2
import os
from tqdm import tqdm
from pathlib import Path

list_roof_camera_ids = [96, 97, 100, 101, 104]

def main(args):
    root = Path(args.capture_root)
    img_dir = root / "images"
    cam_dir = root / "cameras"
    img_undis_dir = root / "images_undistort"

    cameras = dict(np.load(str(cam_dir / "rgb_cameras.npz")))
    os.makedirs(img_undis_dir, exist_ok=True)

    for i in range(len(cameras["ids"])):
        cam_id = cameras["ids"][i]
        if cam_id in list_roof_camera_ids:
            continue
        intrinsic = cameras["intrinsics"][i]
        dist_coeffs = cameras["dist_coeffs"][i] 

        # Find all images in which we want to detect 2D keypoints.
        img_paths = sorted(glob.glob(f"{img_dir}/{cam_id}/*.jpg"))
        print(f"Processing camera {cam_id}: {len(img_paths)} frames")
        os.makedirs(img_undis_dir/ str(cam_id), exist_ok=True)

        for img_path in img_paths:
            img = cv2.imread(img_path)
            # undistort before performing detection 
            img = cv2.undistort(img, intrinsic, dist_coeffs)
            cv2.imwrite(f"{img_undis_dir}/{cam_id}/{Path(img_path).name}", img)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture_root', required=True, help='Path to where capture data is stored.')
    main(parser.parse_args())