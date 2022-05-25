"""
Adapted from extract_video.py (https://github.com/zju3dv/EasyMocap/blob/master/scripts/preprocess/extract_video.py)
"""
import argparse
from mimetypes import init
import cv2
import json
import numpy as np
import os
import lzma
import pickle
import platform

from tqdm import tqdm


def extract_2d(image_paths, keypoints_path, handface, openposefolder="C:\\openpose\\"):
    skip = False
    if os.path.exists(keypoints_path):
        # check the number of images and keypoints
        if len(os.listdir(image_paths)) == len(os.listdir(keypoints_path)):
            skip = True
    if not skip:
        os.makedirs(keypoints_path, exist_ok=True)
        if platform.system() == "Windows":
            cmd_base = 'bin\\OpenPoseDemo.exe'
        else:
            cmd_base = './build/examples/openpose/openpose.bin'
        cmd = '{} --image_dir {} --write_json {} --display 0'.format(cmd_base, image_paths, keypoints_path)
        # cmd = cmd + ' --net_resolution -1x{}'.format(368)
        cmd = cmd + ' --net_resolution 720x480'
        if handface:
            cmd = cmd + ' --hand --face'
        cmd = cmd + ' --render_pose 0'
        os.chdir(openposefolder)
        os.system(cmd)


def read_json(path):
    with open(path) as f:
        data = json.load(f)
    return data

def load_openpose(opname):
    mapname = {'face_keypoints_2d': 'face2d', 'hand_left_keypoints_2d': 'handl2d', 'hand_right_keypoints_2d': 'handr2d'}
    assert os.path.exists(opname), opname
    data = read_json(opname)
    out = []
    pid = 0
    for i, d in enumerate(data['people']):
        keypoints = d['pose_keypoints_2d']
        keypoints = np.array(keypoints).reshape(-1, 3)
        annot = {
            'personID': pid + i,
            'keypoints': keypoints.tolist(),
        }
        for key in ['face_keypoints_2d', 'hand_left_keypoints_2d', 'hand_right_keypoints_2d']:
            if len(d[key]) == 0:
                continue
            kpts = np.array(d[key]).reshape(-1, 3)
            annot[mapname[key]] = kpts.tolist()
        out.append(annot)
    return out


def convert_from_openpose(src, dst, sub, keypoints):
    # convert the 2d pose from openpose
    inputlist = sorted(os.listdir(os.path.join(src)))
    
    predicts = []
    for inp in tqdm(inputlist, desc='{:10s}'.format(os.path.basename(dst))):
        annot = load_openpose(os.path.join(src, inp))
        pose2d = np.asarray(annot[0]["keypoints"])
        predicts.append(pose2d)
    
    predicts = np.expand_dims(np.asarray(predicts), axis=1)
    keypoints[str(sub)] = predicts

def main(args):
    root = args.capture_root

    image_root = os.path.join(root, "images_undistort")
    annot_root = os.path.join(root, "pose2d")

    subs = sorted(os.listdir(image_root))
    if not os.path.exists(annot_root):
        os.makedirs(annot_root, exist_ok=True)

    keypoints = {}
    for sub in subs:
        
        extract_2d(os.path.join(image_root, sub),
                    os.path.join(root, 'openpose', sub),
                    args.handface,
                    openposefolder=args.openposefolder)
        convert_from_openpose(
            src=os.path.join(root, 'openpose', sub),
            dst=annot_root,
            sub=sub,
            keypoints=keypoints
        )

    np.savez(f"{annot_root}/keypoints.npz", **keypoints)
    print(f"2D keypoints detection saved to {annot_root}/keypoints.npz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture_root', required=True)
    parser.add_argument('--openposefolder', required=True, help='Where Openpose binaries are stored.')
    parser.add_argument('--handface', action='store_true', help='If we should detect hand and face keypoints.')
    main(parser.parse_args())