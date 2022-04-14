import os
import glob
import hydra
import cv2
import numpy as np
import torch
from lib.utils import rend_util
# from smplx import SMPL


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
            img = img.reshape(-1, 3)
            img = torch.from_numpy(img).float()
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
            mask = mask.reshape(-1)
            mask = torch.from_numpy(mask).bool()
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
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        if self.num_sample > 0:
            sample_idx = torch.randperm(np.prod(img_size))[:self.num_sample]
            inputs = {
                "object_mask": self.object_masks[idx][sample_idx],
                "uv": uv[sample_idx],
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                # "P": self.P[idx],
                # "C": self.C[idx]
            }
            images = {"rgb": self.images[idx][sample_idx]}
            return inputs, images
        else:
            inputs = {
                "object_mask": self.object_masks[idx],
                "uv": uv,
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                # "P": self.P[idx],
                # "C": self.C[idx],
            }
            images = {"rgb": self.images[idx], "img_size": self.img_sizes[idx]}
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