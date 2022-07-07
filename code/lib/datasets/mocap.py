import os
import glob
import hydra
import cv2
import numpy as np
import pickle as pkl
import torch
from lib.utils import rend_util
def uniform_sampling(data, img_size, num_sample):
    indices = np.random.permutation(np.prod(img_size))[:num_sample]
    output = {
        key: val.reshape(-1, *val.shape[2:])[indices]
        for key, val in data.items()
    }
    return output


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


def uniform_sampling_continuous(data, img_size, num_sample):
    indices = np.random.rand(num_sample, 2)
    indices *= (img_size[0] - 1, img_size[1] - 1)

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

def get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max):
    samples_uniform_row = samples_uniform[:, 0]
    samples_uniform_col = samples_uniform[:, 1]
    index_outside = np.where((samples_uniform_row < bbox_min[0]) | (samples_uniform_row > bbox_max[0]) | (samples_uniform_col < bbox_min[1]) | (samples_uniform_col > bbox_max[1]))[0]
    return index_outside

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

    # get indices for uniform samples outside of bbox
    index_outside = get_index_outside_of_bbox(samples_uniform, bbox_min, bbox_max) + num_sample_bbox

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
    
    # debug
    # img = data['rgb'][:,:,[2,1,0]].copy() * 255.
    # for j in range(0, num_sample):    
    #     pix = output['uv'].astype(np.int32)[j]
    #     if j in index_outside:
    #         output_img = cv2.circle(img, tuple(pix.astype(np.int32)), 3, (0,255,255), -1)
    #     else:
    #         output_img = cv2.circle(img, tuple(pix.astype(np.int32)), 3, (0,0,255), -1)
    #     cv2.imwrite('/home/chen/Desktop/sampling_debug.png', output_img)
    # import ipdb
    # ipdb.set_trace()

    return output, index_outside

class MoCapDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        root = os.path.join("../data", opt.data_dir)
        root = hydra.utils.to_absolute_path(root)

        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        target_cam_ids = np.loadtxt(os.path.join(root, "target_cam_ids.txt"))
        self.num_cam = len(camera_dict) // 2
        self.num_frames = 1
        self.start_frame = 20
        # import pdb
        # pdb.set_trace()
        self.images, self.img_sizes = [], []
        self.object_masks = []
        self.normals = []
        self.bg_images = []
        normalize_rgb = False
        keys = list(dict(camera_dict).keys())

        for cam in range(self.num_cam):
            # images
            img_dir = os.path.join(root, f"image_cam{int(target_cam_ids[cam]):03d}")
            img_paths = sorted(glob.glob(f"{img_dir}/*"))[self.start_frame:self.start_frame+self.num_frames]
            
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

            # masks
            mask_dir = os.path.join(root, f"mask_cam{int(target_cam_ids[cam]):03d}")
            mask_paths = sorted(glob.glob(f"{mask_dir}/*"))[self.start_frame:self.start_frame+self.num_frames]
            
            for i, mask_path in enumerate(mask_paths):
                mask = cv2.imread(mask_path)
                assert mask.shape[:2] == self.img_sizes[
                    i], "Mask image imcompatible with RGB"

                # preprocess: BGR -> Gray -> Mask -> Tensor
                mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 127
                self.object_masks.append(mask)
        
            # normals
            # normal_dir = os.path.join(root, f"normal_cam{int(target_cam_ids[cam]):03d}")
            # normal_paths = sorted(glob.glob(f"{normal_dir}/*"))[self.start_frame:self.start_frame+self.num_frames]

            # for i, normal_path in enumerate(normal_paths):
            #     normal = cv2.imread(normal_path)
            #     assert normal.shape[:2] == self.img_sizes[
            #         i], "Normal image imcompatible with RGB"
            #     normal = ((normal / 255.0)[:, :, ::-1] - 0.5) / 0.5
            #     self.normals.append(normal)
            
            # bg_images
            # bg_img_dir = os.path.join(root, f"bg_image_cam{int(target_cam_ids[cam]):03d}")
            # bg_img_paths = sorted(glob.glob(f"{bg_img_dir}/*"))[self.start_frame:self.start_frame+self.num_frames]

            # for i, bg_img_path in enumerate(bg_img_paths):
            #     bg_img = cv2.imread(bg_img_path)
            #     assert bg_img.shape[:2] == self.img_sizes[
            #         i], "Background image imcompatible with RGB"
                
            #     # preprocess: BGR -> RGB -> Normalize
            #     if normalize_rgb:
            #         bg_img = ((bg_img[:, :, ::-1] / 255) - 0.5) * 2
            #     else:
            #         bg_img = bg_img[:, :, ::-1] / 255
            #     self.bg_images.append(bg_img)
        # SMPL
        # smpl_dir = os.path.join(root, "smpl")
        # smpl_paths = sorted(glob.glob(f"{smpl_dir}/*"))
        # self.smpl_params = []
        # for smpl_path in smpl_paths:
        #     smpl_params = pkl.load(open(smpl_path, "rb"))
        self.shape = np.load(os.path.join(root, "mean_shape.npy"))
        self.poses = np.load(os.path.join(root, 'poses.npy'))[self.start_frame:self.start_frame+self.num_frames]
        self.trans = np.load(os.path.join(root, 'normalize_trans.npy'))[self.start_frame:self.start_frame+self.num_frames]
        # cameras
        cameras = dict(np.load(os.path.join(root, "cameras.npz")))
        # self.P, self.C = [], []
        # for i in range(self.num_cam):
        #     P = cameras[f"cam_{int(target_cam_ids[cam])}"].astype(np.float32)
        #     self.P.append(P)

        #     C = -np.linalg.solve(P[:3, :3], P[:3, 3])
        #     self.C.append(C)

        
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.num_cam)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.num_cam)]

        self.scale = 1 / scale_mats[0][0, 0]

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
        self.sampling_strategy = "weighted"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_size = self.img_sizes[idx]

        uv = np.mgrid[:img_size[0], :img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float() # 1

        smpl_params[1:4] = torch.from_numpy(self.trans[idx%self.num_frames]).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx%self.num_frames]).float()
        smpl_params[76:] = torch.from_numpy(self.shape).float()

        if self.num_sample > 0:
            data = {
                "rgb": self.images[idx],
                "uv": uv,
                "object_mask": self.object_masks[idx],
                # "normal": self.normals[idx],
                # "bg_image": self.bg_images[idx],
            }
            if self.sampling_strategy == "uniform":
                samples = uniform_sampling(data, img_size, self.num_sample)
            elif self.sampling_strategy == "uniform_continuous":
                samples = uniform_sampling_continuous(data, img_size,
                                                      self.num_sample)
            elif self.sampling_strategy == "weighted":
                samples, index_outside = weighted_sampling(data, img_size, self.num_sample)
            inputs = {
                "object_mask": samples["object_mask"].astype(bool),
                # "normal": samples["normal"].astype(np.float32),
                "uv": samples["uv"].astype(np.float32),
                # 'bg_image': samples["bg_image"].astype(np.float32),
                # "P": self.P[idx//self.num_frames],
                # "C": self.C[idx//self.num_frames],
                "intrinsics": self.intrinsics_all[idx//self.num_frames],
                "pose": self.pose_all[idx//self.num_frames],
                "smpl_params": smpl_params,
                'index_outside': index_outside,
                'idx': idx
            }
            images = {"rgb": samples["rgb"].astype(np.float32)}
            return inputs, images
        else:
            inputs = {
                "object_mask": self.object_masks[idx].reshape(-1).astype(bool),
                "uv": uv.reshape(-1, 2).astype(np.float32),
                # "bg_image": self.bg_images[idx].reshape(-1, 3).astype(np.float32),
                # "P": self.P[idx//self.num_frames],
                # "C": self.C[idx//self.num_frames],
                "intrinsics": self.intrinsics_all[idx//self.num_frames],
                "pose": self.pose_all[idx//self.num_frames],
                "smpl_params": smpl_params,
                'idx': idx
            }
            images = {
                "rgb": self.images[idx].reshape(-1, 3).astype(np.float32),
                # "normal": self.normals[idx].reshape(-1, 3).astype(np.float32),
                "img_size": self.img_sizes[idx]
            }
            return inputs, images

class MoCapValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = MoCapDataset(opt)
        image_id = opt.image_id

        self.img_size = self.dataset.img_sizes[image_id]

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = opt.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = int(np.random.choice(len(self.dataset), 1))  
        self.data = self.dataset[image_id]
        inputs, images = self.data
        inputs = {
            "object_mask": inputs["object_mask"],
            "uv": inputs["uv"],
            # "bg_image": inputs["bg_image"],
            # "P": inputs["P"],
            # "C": inputs["C"],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            "smpl_params": inputs["smpl_params"],
            'idx': inputs['idx']
        }
        images = {
            "rgb": images["rgb"],
            # "normal": images["normal"],
            "img_size": images["img_size"],
            'pixel_per_batch': self.pixel_per_batch,
            'total_pixels': self.total_pixels
        }
        return inputs, images

class MoCapTestDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = MoCapDataset(opt)
        self.img_size = self.dataset.img_sizes[0]

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = opt.pixel_per_batch
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # idx = idx + 30
        data = self.dataset[idx]

        inputs, images = data
        inputs = {
            "object_mask": inputs["object_mask"],
            "uv": inputs["uv"],
            # "bg_image": inputs["bg_image"],
            # "P": inputs["P"],
            # "C": inputs["C"],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            "smpl_params": inputs["smpl_params"],
            'idx': inputs['idx']
        }
        images = {
            "rgb": images["rgb"],
            # "normal": images["normal"],
            "img_size": images["img_size"]
        }
        return inputs, images, self.pixel_per_batch, self.total_pixels, idx