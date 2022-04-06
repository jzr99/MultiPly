import os
import glob
import hydra
import cv2
import numpy as np
import torch


class DTUDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        root = os.path.join("../data", opt.data_dir, f"scan{opt.scan_id}")
        root = hydra.utils.to_absolute_path(root)

        # images
        img_dir = os.path.join(root, "image")
        img_paths = sorted(glob.glob(f"{img_dir}/*"))
        self.images, self.img_sizes = [], []
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img_size = img.shape[:2]
            self.img_sizes.append(img_size)

            # preprocess: BGR -> RGB -> Normalize
            img = ((img[:, :, ::-1] / 255) - 0.5) * 2
            img = img.reshape(-1, 3)
            img = torch.from_numpy(img).float()
            self.images.append(img)

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
        cameras = dict(np.load(os.path.join(root, "cameras.npz")))
        self.P, self.C = [], []
        for i in range(len(self.images)):
            P = cameras[f"world_mat_{i}"] @ cameras[f"scale_mat_{i}"]
            self.P.append(P)

            C = -np.linalg.solve(P[:3, :3], P[:3, 3])
            self.C.append(C)

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
                "P": self.P[idx],
                "C": self.C[idx],
            }
            images = {"rgb": self.images[idx][sample_idx]}
            return inputs, images
        else:
            inputs = {
                "object_mask": self.object_masks[idx],
                "uv": uv,
                "P": self.P[idx],
                "C": self.C[idx],
            }
            images = {"rgb": self.images[idx], "img_size": self.img_sizes[idx]}
            return inputs, images


class DTUValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        dataset = DTUDataset(opt)
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
            "P": inputs["P"],
            "C": inputs["C"]
        }
        images = {
            "rgb": images["rgb"][indices],
            "img_size": images["img_size"]
        }
        return inputs, images