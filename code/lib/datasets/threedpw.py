from operator import index
import os
import glob
from unittest import skip
import hydra
import cv2
import numpy as np
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

class ThreeDPWDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        root = os.path.join("../data", opt.data_dir)
        root = hydra.utils.to_absolute_path(root)

        self.start_frame = opt.start_frame # 0 # 11 # 94
        self.end_frame = opt.end_frame # 5 # 658 # 462 # 231 # 204
        self.skip_step = 1
        self.images, self.img_sizes = [], []
        self.object_masks = []
        self.training_indices = list(range(opt.start_frame, opt.end_frame, self.skip_step))

        if opt.exclude_frames != 'none':
            for i in opt.exclude_frames:
                self.training_indices.remove(i)

        self.bg_images = []
        # self.parsing_masks = []

        self.normalize_rgb = False

        # images
        img_dir = os.path.join(root, "image")
        self.img_paths = sorted(glob.glob(f"{img_dir}/*.png"))
        self.img_paths = [self.img_paths[i] for i in self.training_indices]
        self.img_size = cv2.imread(self.img_paths[0]).shape[:2]
        # for img_path in img_paths[self.training_indices]: # [::self.skip_step]:
        #     img = cv2.imread(img_path)
        #     img_size = img.shape[:2]
        #     self.img_sizes.append(img_size)

        #     # preprocess: BGR -> RGB -> Normalize
        #     if self.normalize_rgb:
        #         img = ((img[:, :, ::-1] / 255) - 0.5) * 2
        #     else:
        #         img = img[:, :, ::-1] / 255
        #     self.images.append(img)
        self.n_images = len(self.img_paths)
        # masks
        mask_dir = os.path.join(root, "mask")
        self.mask_paths = sorted(glob.glob(f"{mask_dir}/*.png"))
        self.mask_paths = [self.mask_paths[i] for i in self.training_indices]
        # for i, mask_path in enumerate(mask_paths[self.training_indices]): # [::self.skip_step]:
        #     mask = cv2.imread(mask_path)
        #     assert mask.shape[:2] == self.img_sizes[
        #         i], "Mask image imcompatible with RGB"

        #     # preprocess: BGR -> Gray -> Mask -> Tensor
        #     mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0
        #     self.object_masks.append(mask)
        
        # ground_masks only for scenes with strong shadows
        # ground_mask_dir = os.path.join(root, "ground_mask")
        # ground_mask_paths = sorted(glob.glob(f"{ground_mask_dir}/*"))
        # for i, ground_mask_path in enumerate(ground_mask_paths[self.training_indices]): # [::self.skip_step]):
        #     ground_mask = cv2.imread(ground_mask_path)
        #     assert ground_mask.shape[:2] == self.img_sizes[
        #         i], "Ground mask image imcompatible with RGB"
            
        #     # preprocess: BGR -> Gray -> Mask -> Tensor
        #     ground_mask = cv2.cvtColor(ground_mask, cv2.COLOR_BGR2GRAY) > 0
        #     self.ground_masks.append(ground_mask)
        
        # normals
        # normal_dir = os.path.join(root, "normal")
        # normal_paths = sorted(glob.glob(f"{normal_dir}/*.png"))
        
        # for i, normal_path in enumerate(normal_paths[self.training_indices]):
        #     normal = cv2.imread(normal_path)
        #     assert normal.shape[:2] == self.img_sizes[
        #         i], "Normal image imcompatible with RGB"
        #     normal = ((normal / 255.0)[:, :, ::-1] - 0.5) / 0.5
        #     # normal = normal / np.linalg.norm(normal, axis=-1)[:, :, None]
        #     self.normals.append(normal)

        # body parsing
        # parsing_dir = os.path.join(root, "body_parsing")
        # parsing_paths = sorted(glob.glob(f"{parsing_dir}/*"))
        # for i, parsing_path in enumerate(parsing_paths[::self.skip_step]):
        #     parsing = cv2.imread(parsing_path)[..., 0]
        #     assert parsing.shape[:2] == self.img_sizes[i], "Parsing image imcompatible with RGB"
        #     self.parsing_masks.append(parsing)

        # bg_images
        # bg_img_dir = os.path.join(root, "bg_partial")
        # bg_img_paths = sorted(glob.glob(f"{bg_img_dir}/*"))
        # for i, bg_img_path in enumerate(bg_img_paths[::self.skip_step]):
        #     bg_img = cv2.imread(bg_img_path)
        #     assert bg_img.shape[:2] == self.img_sizes[i], "Background image imcompatible with RGB"
            
        #     # preprocess: BGR -> RGB -> Normalize
        #     if self.normalize_rgb:
        #         bg_img = ((bg_img[:, :, ::-1] / 255) - 0.5) * 2
        #     else:
        #         bg_img = bg_img[:, :, ::-1] / 255
        #     self.bg_images.append(bg_img)

        self.shape = np.load(os.path.join(root, "mean_shape.npy"))
        self.poses = np.load(os.path.join(root, 'poses.npy'))[self.training_indices] # [::self.skip_step]
        self.trans = np.load(os.path.join(root, 'normalize_trans.npy'))[self.training_indices] # [::self.skip_step]
        # cameras
        # cameras = np.load(os.path.join(root, "cameras.npy"))[self.training_indices] # [::self.skip_step]

        self.P, self.C = [], []
        # for i in range(cameras.shape[0]):
        #     P = cameras[i].astype(np.float32)
        #     self.P.append(P)

        #     C = -np.linalg.solve(P[:3, :3], P[:3, 3])
        #     self.C.append(C)

        camera_dict = np.load(os.path.join(root, "cameras_normalize.npz"))
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in self.training_indices] # range(0, self.n_images, self.skip_step)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in self.training_indices] # range(0, self.n_images, self.skip_step)

        self.scale = 1 / scale_mats[0][0, 0]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            self.P.append(P)
            C = -np.linalg.solve(P[:3, :3], P[:3, 3])
            self.C.append(C)
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())
        assert len(self.intrinsics_all) == len(self.pose_all) # == len(self.images)

        # other properties
        self.num_sample = opt.num_sample
        # self.img_sizes = np.array(self.img_sizes)
        self.sampling_strategy = "weighted"

    def __len__(self):
        return self.n_images # len(self.images)

    def load_body_model_params(self):
        body_model_params = {}
        return body_model_params
    def __getitem__(self, idx):
        img = cv2.imread(self.img_paths[idx])
        # preprocess: BGR -> RGB -> Normalize
        if self.normalize_rgb:
            img = ((img[:, :, ::-1] / 255) - 0.5) * 2
        else:
            img = img[:, :, ::-1] / 255

        mask = cv2.imread(self.mask_paths[idx])
        # preprocess: BGR -> Gray -> Mask -> Tensor
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 0

        img_size = self.img_size # [idx]

        uv = np.mgrid[:img_size[0], :img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = torch.from_numpy(np.asarray(self.scale)).float() 

        smpl_params[1:4] = torch.from_numpy(self.trans[idx]).float()
        smpl_params[4:76] = torch.from_numpy(self.poses[idx]).float()
        smpl_params[76:] = torch.from_numpy(self.shape).float()

        if self.num_sample > 0:
            data = {
                "rgb": img, # self.images[idx],
                "uv": uv,
                "object_mask": mask, # self.object_masks[idx],
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
            # samples["normal"] = samples["normal"] / np.linalg.norm(samples["normal"], axis=-1)[:, None]
            inputs = {
                # "normal": samples["normal"].astype(np.float32),
                "uv": samples["uv"].astype(np.float32),
                # 'bg_image': samples["bg_image"].astype(np.float32),
                "P": self.P[idx],
                "C": self.C[idx],
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                'index_outside': index_outside,
                "idx": idx
            }
            images = {"rgb": samples["rgb"].astype(np.float32)}
            return inputs, images
        else:
            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                # "bg_image": self.bg_images[idx].reshape(-1, 3).astype(np.float32),
                "P": self.P[idx],
                "C": self.C[idx],
                "intrinsics": self.intrinsics_all[idx],
                "pose": self.pose_all[idx],
                "smpl_params": smpl_params,
                "idx": idx
            }
            images = {
                "rgb": img.reshape(-1, 3).astype(np.float32), # self.images[idx].reshape(-1, 3).astype(np.float32),
                # "normal": self.normals[idx].reshape(-1, 3).astype(np.float32),
                "img_size": self.img_size # [idx]
            }
            return inputs, images

class ThreeDPWValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        self.dataset = ThreeDPWDataset(opt)
        image_id = opt.image_id
        self.img_size = self.dataset.img_size # img_sizes[image_id]

        self.total_pixels = np.prod(self.img_size)
        self.pixel_per_batch = opt.pixel_per_batch

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        image_id = int(np.random.choice(len(self.dataset), 1))  
        self.data = self.dataset[image_id]
        inputs, images = self.data

        inputs = {
            "uv": inputs["uv"],
            # "bg_image": inputs["bg_image"],
            "P": inputs["P"],
            "C": inputs["C"],
            "intrinsics": inputs['intrinsics'],
            "pose": inputs['pose'],
            "smpl_params": inputs["smpl_params"],
            'image_id': image_id,
            "idx": inputs['idx']
        }
        images = {
            "rgb": images["rgb"],
            # "normal": images["normal"],
            "img_size": images["img_size"],
            'pixel_per_batch': self.pixel_per_batch,
            'total_pixels': self.total_pixels
        }
        return inputs, images

class ThreeDPWTestDataset(torch.utils.data.Dataset):
    def __init__(self, opt, free_view_render=False, canonical_vis=False, animation_path='/home/chen/RGB-PINA/data/bike'):
        self.free_view_render = free_view_render
        self.canonical_vis = canonical_vis
        self.animation = True if animation_path is not None else False
        self.dataset = ThreeDPWDataset(opt)
        if self.free_view_render:
            start = 0
            steps = 60
            step_size = 6
            self.new_poses = []
            self.image_id = 154
            self.data = self.dataset[self.image_id]
            self.img_size = self.dataset.img_size # [self.image_id]
            self.total_pixels = np.prod(self.img_size)
            self.pixel_per_batch = opt.pixel_per_batch
            target_inputs, images = self.data
            from scipy.spatial.transform import Rotation as scipy_R
            for i in range(steps):
                rotation_angle_y = start+i*(step_size)
                pose = target_inputs['pose'].clone()
                new_pose = rend_util.get_new_cam_pose_fvr(pose, rotation_angle_y)
                self.new_poses.append(new_pose)
        else:
            self.img_size = self.dataset.img_size # [0]

            self.total_pixels = np.prod(self.img_size)
            self.pixel_per_batch = opt.pixel_per_batch
            if self.animation:
                self.test_indices = [2, 7, 12, 17, 22, 27, 32, 37, 42, 47]
                if self.test_indices is not None:
                    self.animation_poses = np.load(os.path.join(animation_path, 'poses.npy'))[self.test_indices]
                else:
                    self.animation_poses = np.load(os.path.join(animation_path, 'poses.npy'))
                # self.animation_transl = np.load(os.path.join(animation_path, 'opt_transl.npy'))
    def __len__(self):
        if self.free_view_render:
            return len(self.new_poses)
        elif self.animation:
            return self.animation_poses.shape[0]
        else:
            return len(self.dataset)

    def __getitem__(self, idx):
        idx == 941
        # manually set index
        # idx += 422
        # if idx == len(self.dataset) - 1:
        #     idx = len(self.dataset) - 1
        if self.free_view_render:
            uv = np.mgrid[:self.img_size[0], :self.img_size[1]].astype(np.int32)
            uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)
            target_inputs, images = self.data

            inputs = {
                "uv": uv.reshape(-1, 2).astype(np.float32),
                "intrinsics": target_inputs['intrinsics'],
                "pose": self.new_poses[idx], # target_inputs['pose'], # self.pose_all[idx],
                'P': target_inputs['P'],
                'C': target_inputs['C'],
                "smpl_params": target_inputs["smpl_params"],
                'image_id': self.image_id
            }
            images = {
                    "img_size": self.img_size}
            if self.canonical_vis:
                cano_smpl_params = inputs["smpl_params"].clone()
                # cano_smpl_params[1:4] = torch.tensor([ 0.0117,  0.2990, -0.0060]).float()
                cano_smpl_params[4:76] = 0 
                cano_smpl_params[9] = np.pi/6
                cano_smpl_params[12] = -np.pi/6
                inputs.update({'smpl_params': cano_smpl_params})
            if self.animation:
                data = self.dataset[0]
                inputs, images = data
        elif self.animation:
            # data = self.dataset[0]
            data = self.dataset[self.test_indices[idx]]

            inputs, images = data
            inputs = {
                "uv": inputs["uv"],
                # "bg_image": inputs["bg_image"],
                "P": inputs["P"],
                "C": inputs["C"],
                "intrinsics": inputs['intrinsics'],
                "pose": inputs['pose'],
                "smpl_params": inputs["smpl_params"],
                "idx": inputs['idx']
            }
            anim_smpl_pose = torch.from_numpy(self.animation_poses[idx]).float()
            # anim_smpl_transl = torch.from_numpy(self.animation_transl[idx]).float()
            anim_smpl_params = inputs["smpl_params"].clone()
            # anim_smpl_params[1:4] = anim_smpl_transl
            anim_smpl_params[4:76] = torch.tensor(anim_smpl_pose).float()
            inputs.update({'smpl_params': anim_smpl_params})
            inputs.update({'image_id': 0})
        else:
            data = self.dataset[idx]

            inputs, images = data
            inputs = {
                "uv": inputs["uv"],
                # "bg_image": inputs["bg_image"],
                "P": inputs["P"],
                "C": inputs["C"],
                "intrinsics": inputs['intrinsics'],
                "pose": inputs['pose'],
                "smpl_params": inputs["smpl_params"],
                "idx": inputs['idx']
            }
            images = {
                "rgb": images["rgb"],
                # "normal": images["normal"],
                "img_size": images["img_size"]
            }
            if self.canonical_vis:
                cano_smpl_params = inputs["smpl_params"].clone()
                cano_smpl_params[4:76] = 0 
                cano_smpl_params[9] = np.pi/6
                cano_smpl_params[12] = -np.pi/6
                cano_smpl_params
                inputs.update({'smpl_params': cano_smpl_params})
        return inputs, images, self.pixel_per_batch, self.total_pixels, idx, self.free_view_render, self.canonical_vis, self.animation