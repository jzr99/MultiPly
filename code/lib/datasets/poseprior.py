import os
import glob
import hydra
import cv2
import numpy as np
import pickle as pkl
import torch

class PosePriorDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        root = os.path.join("../data", opt.data_dir)
        root = hydra.utils.to_absolute_path(root)

        dataset_path = '/home/chen/disk2/AMASS/Initialization_Avatar/split'
        minimal_body_path = '/home/chen/Desktop/v_template.npy'

        self.verts_body = np.load(minimal_body_path)

        self.file_list = sorted(glob.glob(os.path.join(dataset_path, '*.pkl')))
        self.file_list = self.file_list[::100]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        poses = pkl.load(open(self.file_list[idx], 'rb'))['poses']

        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = 1
        # global translation and rotation are excluded
        # Last 6 parameters in poses correspond to SMPL-H hands
        smpl_params[7:70] = torch.tensor(poses[3:66]).float()
        


        inputs = {

            "smpl_params": smpl_params
        }

        return inputs



class PosePriorValDataset(torch.utils.data.Dataset):
    def __init__(self, opt):
        dataset = PosePriorDataset(opt)
        image_id = opt.image_id

        self.data = dataset[image_id]

    def __len__(self):
        # return (self.total_pixels + self.pixel_per_batch -
        #         1) // self.pixel_per_batch
        return 1

    def __getitem__(self, idx):
        indices = list(
            range(idx * self.pixel_per_batch,
                  min((idx + 1) * self.pixel_per_batch, self.total_pixels)))

        inputs = self.data
        inputs = {
            "smpl_params": inputs["smpl_params"]
        }

        return inputs