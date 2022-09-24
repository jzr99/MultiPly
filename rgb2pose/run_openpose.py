# From Python
# It requires OpenCV installed for Python
import sys
import cv2
import os
import numpy as np
from sys import platform
import argparse
import time
from sklearn.neighbors import NearestNeighbors
def crop_image(img, bbox, batch=False):
    if batch:
        return img[:, int(bbox[1]):(int(bbox[1]) + int(bbox[3] - bbox[1])), int(bbox[0]): (int(bbox[0]) + int(bbox[2] - bbox[0]))]
    else:
        return img[int(bbox[1]):(int(bbox[1]) + int(bbox[3] - bbox[1])), int(bbox[0]): (int(bbox[0]) + int(bbox[2] - bbox[0]))]

def recover_cropped_img(cropped_img, bbox, W, H):
    img = np.zeros((H, W, 3))
    img[int(bbox[1]):(int(bbox[1]) + int(bbox[3] - bbox[1])), int(bbox[0]): (int(bbox[0]) + int(bbox[2] - bbox[0]))] = cropped_img
    return img

def recover_cropped_joints(joints_cropped, bbox):
    joints = np.zeros(joints_cropped.shape)
    joints[:, 0] = joints_cropped[:, 0] + int(bbox[0]) # int(bbox[1])
    joints[:, 1] = joints_cropped[:, 1] + int(bbox[1]) # int(bbox[0])
    joints[:, 2] = joints_cropped[:, 2]
    return joints

def read_img(img_path, mask_path):
    _img = cv2.imread(img_path) #  Image.open(img_path).convert('RGB')  # return is RGB pic
    W, H = _img.shape[1], _img.shape[0]

    mask = cv2.imread(mask_path)[:, :, 0]
    where = np.asarray(np.where(mask))
    bbox_min = where.min(axis=1)
    bbox_min = bbox_min - 25
    bbox_max = where.max(axis=1)
    bbox_max = bbox_max + 25
    left, top, right, bottom = bbox_min[1], bbox_min[0], bbox_max[1], bbox_max[
        0]
    # if right - left < 500:
    #     padding = (500 -(right-left))//2
    #     left = left - padding
    #     right = right + padding
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, W)
    bottom = min(bottom, H)
    crop_bbox = (left, top, right, bottom)
    bbox_center = np.array([left + (right - left) / 2, top + (bottom - top) / 2])
    # _img_crop = _img.crop(crop_bbox)
    _img_crop = crop_image(_img, crop_bbox)
    # cv2.imwrite('/home/chen/Desktop/_img_crop.png', _img_crop)

    return _img_crop.copy(), crop_bbox, W, H, bbox_center

try:
    # Import Openpose (Windows/Ubuntu/OSX)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Change these variables to point to the correct folder (Release/x64 etc.)
    sys.path.append('/home/chen/openpose/build/python')
    # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
    # sys.path.append('/usr/local/python')
    from openpose import pyopenpose as op

    # Flags
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="/home/chen/disk2/Youtube_Videos/outdoors_fencing_01/frames", help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    parser.add_argument("--no_display", default=False, help="Enable to disable the visual display.")
    args = parser.parse_known_args()

    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "/home/chen/openpose/models/"
    # params["face"] = True
    # params["hand"] = True
    params['scale_number'] = 1
    params['scale_gap'] = 0.25
    params['net_resolution'] = '720x480' # 1312x736 720x480

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()

    # Read frames on directory
    imagePaths = op.get_images_on_directory(args[0].image_dir)
    import glob
    maskPaths = sorted(glob.glob(f'{args[0].image_dir}/../init_mask/*.png'))
    start = time.time()

    if not os.path.exists(f'{args[0].image_dir}/../openpose'):
        os.makedirs(f'{args[0].image_dir}/../openpose')

    # Process and display images
    nbrs = NearestNeighbors(n_neighbors=1)
    for idx, imagePath in enumerate(imagePaths):
        datum = op.Datum()
        imageToProcess = cv2.imread(imagePath)
        maskPath = maskPaths[idx]
        _, _, _, _, bbox_center = read_img(imagePath, maskPath)
        datum.cvInputData = imageToProcess
        opWrapper.emplaceAndPop(op.VectorDatum([datum]))

        # print("Body keypoints: \n" + str(datum.poseKeypoints))
        poseKeypoints = datum.poseKeypoints
        # poseKeypoints = recover_cropped_joints(poseKeypoints, crop_bbox)

        nbrs.fit(poseKeypoints[:, 8, :2])

        actor = nbrs.kneighbors(bbox_center.reshape(1, -1), return_distance=False).ravel()[0]
        poseKeypoints = poseKeypoints[actor]
        np.save(f'{args[0].image_dir}/../openpose/%04d.npy' % idx, poseKeypoints)
        for jth in range(0, poseKeypoints.shape[0]):
            output_img = cv2.circle(imageToProcess, tuple(poseKeypoints.astype(np.int32)[jth, :2]), 3, (0,0,255), -1)
        cv2.imwrite(f'{args[0].image_dir}/../openpose/%04d.png' % idx, output_img)

        # cv2.imwrite('/home/chen/disk2/Youtube_Videos/Easy_on_me/openpose/%04d.png' % idx, datum.cvOutputData)
    end = time.time()
    print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
except Exception as e:
    print(e)
    sys.exit(-1)
