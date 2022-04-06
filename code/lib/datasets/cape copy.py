import os
import glob
import hydra
import cv2
import numpy as np
import pickle as pkl
import torch
# from smplx import SMPL


class CapeDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        root = os.path.join("../data", opt.data_dir)
        root = hydra.utils.to_absolute_path(root)
        self.num_cam = 1
        self.num_frames = 161 # 112//2 
        # import pdb
        # pdb.set_trace()
        self.images, self.img_sizes = [], []
        self.object_masks = []
        for cam in range(self.num_cam):
            # images
            img_dir = os.path.join(root, f"image_cam{cam:02d}")
            img_paths = sorted(glob.glob(f"{img_dir}/*"))
            
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
            mask_dir = os.path.join(root, f"mask_cam{cam:02d}")
            mask_paths = sorted(glob.glob(f"{mask_dir}/*"))
            
            for i, mask_path in enumerate(mask_paths):
                mask = cv2.imread(mask_path)
                assert mask.shape[:2] == self.img_sizes[
                    i], "Mask image imcompatible with RGB"

                # preprocess: BGR -> Gray -> Mask -> Tensor
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 127
                mask = mask.reshape(-1)
                mask = torch.from_numpy(mask).bool()
                self.object_masks.append(mask)

        # SMPL
        # smpl_dir = os.path.join(root, "smpl")
        # smpl_paths = sorted(glob.glob(f"{smpl_dir}/*"))
        # self.smpl_params = []
        # for smpl_path in smpl_paths:
        #     smpl_params = pkl.load(open(smpl_path, "rb"))
        self.poses = np.load(os.path.join(root, 'poses.npy'))
        self.trans = np.load(os.path.join(root, 'normalize_trans.npy'))
        # cameras
        cameras = dict(np.load(os.path.join(root, "cameras.npz")))
        self.P, self.C = [], []
        for i in range(self.num_cam):
            P = cameras[f"cam_{i}"].astype(np.float32)
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
        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = 1
        smpl_params[1:4] = torch.from_numpy(self.trans[idx%self.num_frames]).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx%self.num_frames]).float()

        if self.num_sample > 0:
            sample_idx = torch.randperm(np.prod(img_size))[:self.num_sample]
            inputs = {
                "object_mask": self.object_masks[idx][sample_idx],
                "uv": uv[sample_idx],
                "P": self.P[idx//self.num_frames],
                "C": self.C[idx//self.num_frames],
                "smpl_params": smpl_params
            }
            images = {"rgb": self.images[idx][sample_idx]}
            return inputs, images
        else:
            inputs = {
                "object_mask": self.object_masks[idx],
                "uv": uv,
                "P": self.P[idx//self.num_frames],
                "C": self.C[idx//self.num_frames],
                "smpl_params": smpl_params
            }
            images = {"rgb": self.images[idx], "img_size": self.img_sizes[idx]}
            return inputs, images

class CapeValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        dataset = CapeDataset(opt)
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
            "C": inputs["C"],
            "smpl_params": inputs["smpl_params"]
        }
        images = {
            "rgb": images["rgb"][indices],
            "img_size": images["img_size"]
        }
        return inputs, images