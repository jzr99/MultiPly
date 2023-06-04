import sys
import cv2
import os
import numpy as np
import argparse
import time
import glob
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
def get_bbox_center(img_path, mask_path):
    _img = cv2.imread(img_path)
    W, H = _img.shape[1], _img.shape[0]

    mask = cv2.imread(mask_path)[:, :, 0]
    where = np.asarray(np.where(mask))
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)
    left, top, right, bottom = bbox_min[1], bbox_min[0], bbox_max[1], bbox_max[
        0]
    left = max(left, 0)
    top = max(top, 0)
    right = min(right, W)
    bottom = min(bottom, H)
    bbox_center = np.array([left + (right - left) / 2, top + (bottom - top) / 2])
    return bbox_center

def get_mask_center(mask_path):
    mask = cv2.imread(mask_path)[:, :, 0]
    where = np.asarray(np.where(mask))
    center = where.mean(axis=1)[::-1]
    return center


def main(args):
    try:
        sys.path.append(args.openpose_dir + '/build/python')
        # we use the python binding of openpose
        from openpose import pyopenpose as op
        DIR = './raw_data'
        # Flags
        params = dict()
        params['model_folder'] = args.openpose_dir + '/models/'
        params['scale_number'] = 1
        params['scale_gap'] = 0.25
        params['net_resolution'] = '720x480'
        # params['net_resolution'] = '360x240'

        # Starting OpenPose
        # print("before Starting OpenPose")
        opWrapper = op.WrapperPython()
        opWrapper.configure(params)
        opWrapper.start()
        # print("after Starting OpenPose")

        # Read frames on directory
        img_dir = f'{DIR}/{args.seq}/frames'
        # this line below will lead to a segfault
        # imagePaths = sorted(glob.glob(f'{img_dir}/*.jpg'))
        imagePaths = sorted(glob.glob(f'{img_dir}/*.png'))
        # imagePaths = op.get_images_on_directory(img_dir)
        maskPaths_0 = sorted(glob.glob(f'{img_dir}/../init_mask/0/*.png'))
        maskPaths_1 = sorted(glob.glob(f'{img_dir}/../init_mask/1/*.png'))
        start = time.time()

        if not os.path.exists(f'{img_dir}/../openpose'):
            os.makedirs(f'{img_dir}/../openpose')

        # Process and display images
        nbrs = NearestNeighbors(n_neighbors=1)
        for idx, imagePath in enumerate(tqdm(imagePaths)):
            if idx < 215:
                continue
            # print("before op.Datum()")
            datum = op.Datum()
            # print("after op.Datum()")
            imageToProcess = cv2.imread(imagePath)
            maskPath_0 = maskPaths_0[idx]
            maskPath_1 = maskPaths_1[idx]
            # bbox_center_0 = get_bbox_center(imagePath, maskPath_0)
            # bbox_center_1 = get_bbox_center(imagePath, maskPath_1)
            mask_center_0 = get_mask_center(maskPath_0)
            mask_center_1 = get_mask_center(maskPath_1)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            poseKeypoints = datum.poseKeypoints
            # print("number of detected person", poseKeypoints.shape)
            center_2D = (poseKeypoints[:, :, :2] * poseKeypoints[:, :, [-1]]).sum(axis=1) / poseKeypoints[:, :, -1].sum(axis=1, keepdims=True)
            if center_2D.shape[0] > 2:
                print("detect more than 2 person, truncating to 2")
                center_2D = center_2D[:2]
            elif center_2D.shape[0] < 2:
                print("detect less than 2 person, ignored")
                np.save(f'{img_dir}/../openpose/%04d.npy' % idx, np.zeros(1))
                continue
            if np.linalg.norm(center_2D[0] - mask_center_0) + np.linalg.norm(center_2D[1] - mask_center_1) > np.linalg.norm(center_2D[0] - mask_center_1) + np.linalg.norm(center_2D[1] - mask_center_0):
                order = [1, 0]
            else:
                order = [0, 1]
            # nbrs.fit(center_2D)
            # nbrs.fit(poseKeypoints[:, 8, :2])

            actor_0 = order[0]
            actor_1 = order[1]
            # actor_0 = nbrs.kneighbors(bbox_center_0.reshape(1, -1), return_distance=False).ravel()[0]
            # actor_0 = nbrs.kneighbors(mask_center_0.reshape(1, -1), return_distance=False).ravel()[0]
            poseKeypoints_0 = poseKeypoints[actor_0]
            # actor_1 = nbrs.kneighbors(mask_center_1.reshape(1, -1), return_distance=False).ravel()[0]
            # actor_1 = nbrs.kneighbors(bbox_center_1.reshape(1, -1), return_distance=False).ravel()[0]
            poseKeypoints_1 = poseKeypoints[actor_1]

            # import pdb; pdb.set_trace()
            np.save(f'{img_dir}/../openpose/%04d.npy' % idx, np.stack([poseKeypoints_0, poseKeypoints_1], axis=0))
            output_img = datum.cvOutputData
            for point in poseKeypoints_0:
                cv2.circle(output_img, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
            for point in poseKeypoints_1:
                cv2.circle(output_img, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
            cv2.circle(output_img, (int(center_2D[actor_0][0]),int(center_2D[actor_0][1])), 10, (0, 0, 255), -1)
            cv2.circle(output_img, (int(center_2D[actor_1][0]),int(center_2D[actor_1][1])), 10, (0, 255, 0), -1)
            cv2.circle(output_img, (int(mask_center_0[0]),int(mask_center_0[1])), 10, (0, 0, 255), 3)
            cv2.circle(output_img, (int(mask_center_1[0]),int(mask_center_1[1])), 10, (0, 255, 0), 3)
            cv2.imwrite(f'{img_dir}/../openpose/%04d.png' % idx, output_img)
        end = time.time()
        print("OpenPose demo successfully finished. Total time: " + str(end - start) + " seconds")
    except Exception as e:
        print(e)
        sys.exit(-1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run OpenPose on a sequence")
    # directory of openpose
    parser.add_argument('--openpose_dir', type=str, help="Directory of openpose")
    # sequence name
    parser.add_argument('--seq', type=str, help="Process a directory of images. Read all standard formats (jpg, png, bmp, etc.).")
    args = parser.parse_args()
    main(args)