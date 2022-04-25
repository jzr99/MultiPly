import os
import glob
import hydra
import cv2
import numpy as np
import torch
from lib.utils import rend_util
# from smplx import SMPL

def bilinear_interpolation(xs, ys, dist_map):
    x1 = np.floor(xs).astype(np.int32)
    y1 = np.floor(ys).astype(np.int32)
    x2 = x1 + 1
    y2 = y1 + 1

    dx = np.expand_dims(np.stack([x2 - xs, xs - x1], axis=1), axis=1)
    dy = np.expand_dims(np.stack([y2 - ys, ys - y1], axis=1), axis=2)
    Q = np.stack([
        dist_map[x1, y1], dist_map[x1, y2], dist_map[x2, y1], dist_map[x2, y2]
    ], axis=1).reshape(-1, 2, 2)
    return np.squeeze(dx @ Q @ dy)  # ((x2 - x1) * (y2 - y1)) = 1

def weighted_sampling(data, img_size, num_sample):
    # calculate bounding box
    mask = data["object_mask"]
    where = np.asarray(np.where(mask))
    bbox_min = where.min(axis=1)
    bbox_max = where.max(axis=1)

    num_sample_bbox = int(num_sample * 0.9)
    samples_bbox = np.random.rand(num_sample_bbox, 2)
    samples_bbox = samples_bbox * (bbox_max - bbox_min) + bbox_min

    num_sample_uniform = num_sample - num_sample_bbox
    samples_uniform = np.random.rand(num_sample_uniform, 2)
    samples_uniform *= (img_size[0] - 1, img_size[1] - 1)

    indices = np.concatenate([samples_bbox, samples_uniform], axis=0)
    output = {}
    for key, val in data.items():
        if len(val.shape) == 3:
            new_val = np.stack([
                bilinear_interpolation(indices[:, 0], indices[:, 1], val[:, :, i])
                for i in range(val.shape[2])
            ], axis=-1)
        else:
            new_val = bilinear_interpolation(indices[:, 0], indices[:, 1], val)
        new_val = new_val.reshape(-1, *val.shape[2:])
        output[key] = new_val
    return output

class SampleDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        root = os.path.join("../data", opt.data_dir)
        root = hydra.utils.to_absolute_path(root)

        # images
        img_dir = os.path.join(root, "image")
        img_paths = sorted(glob.glob(f"{img_dir}/*"))
        self.images, self.img_sizes = [], []

        normalize_rgb = False
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img_size = img.shape[:2]
            self.img_sizes.append(img_size)

            # preprocess: BGR -> RGB -> Normalize
            if normalize_rgb:
                img = ((img[:, :, ::-1] / 255) - 0.5) * 2
            else:
                img = img[:, :, ::-1] / 255
            self.images.append(img)

        self.n_images = len(img_paths)
        # masks
        mask_dir = os.path.join(root, "mask")
        mask_paths = sorted(glob.glob(f"{mask_dir}/*"))
        self.object_masks = []
        for i, mask_path in enumerate(mask_paths):
            mask = cv2.imread(mask_path)
            assert mask.shape[:2] == self.img_sizes[
                i], "Mask image imcompatible with RGB"

            # preprocess: BGR -> Gray -> Mask -> Tensor
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 127
            self.object_masks.append(mask)

        # cameras
        # cameras = dict(np.load(os.path.join(root, "cameras.npz")))
        # self.P, self.C = [], []
        # for i in range(len(self.images)):
        #     P = cameras[f"cam_{i}"].astype(np.float32)
        #     self.P.append(P)

        #     C = -np.linalg.solve(P[:3, :3], P[:3, 3])
        #     self.C.append(C)

        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        # other properties
        self.num_sample = opt.num_sample
        self.img_sizes = np.array(self.img_sizes)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_size = self.img_sizes[idx]

        uv = np.mgrid[:img_size[0], :img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        if self.num_sample > 0:
            data = {
                "rgb": self.images[idx],
                "uv": uv,
                "object_mask": self.object_masks[idx],
            }
            samples = weighted_sampling(data, img_size, self.num_sample)
            # sample_idx = torch.randperm(np.prod(img_size))[:self.num_sample]
            inputs = {
                "object_mask": samples["object_mask"] > 0.5,
                "uv": samples["uv"].astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                # "P": self.P[idx],
                # "C": self.C[idx]
            }
            images = {"rgb": samples["rgb"].astype(np.float32)}
            return inputs, images
        else:
            inputs = {
                "object_mask": self.object_masks[idx].reshape(-1),
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                # "P": self.P[idx],
                # "C": self.C[idx],
            }
            images = {"rgb": self.images[idx].reshape(-1, 3).astype(np.float32), "img_size": self.img_sizes[idx]}
            return inputs, images

class SampleValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        dataset = SampleDataset(opt)
        image_id = opt.image_id

        self.data = dataset[image_id]
        self.img_size = dataset.img_sizes[image_id]

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = opt.pixel_per_batch

    def __len__(self):
        return (self.total_pixels + self.pixel_per_batch -
                1) // self.pixel_per_batch

    def __getitem__(self, idx):
        indices = list(
            range(idx * self.pixel_per_batch,
                  min((idx + 1) * self.pixel_per_batch, self.total_pixels)))

        inputs, images = self.data
        inputs = {
            "object_mask": inputs["object_mask"][indices],
            "uv": inputs["uv"][indices],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            # "P": inputs["P"],
            # "C": inputs["C"]
        }
        images = {
            "rgb": images["rgb"][indices],
            "img_size": images["img_size"]
        }
        return inputs, images