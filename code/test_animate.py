from lib.model.idr import IDR
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob
import cv2
import numpy as np
import torch
from tqdm import trange

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    print("Checkpoint", checkpoint)

    betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.test.data_dir, 'mean_shape.npy')
    idr_model = IDR.load_from_checkpoint(checkpoint, opt=opt, betas_path=betas_path)
    model = idr_model.model
    model.eval()
    model = model.cuda()

    img_sizes = []
    images = []
    object_masks = []

    root = os.path.join("../data", opt.dataset.test.data_dir)
    root = hydra.utils.to_absolute_path(root)
    # images
    img_dir = os.path.join(root, "image")
    img_paths = sorted(glob.glob(f"{img_dir}/*"))

    for img_path in img_paths:
        img = cv2.imread(img_path)
        img_size = img.shape[:2]
        img_sizes.append(img_size)

        # preprocess: BGR -> RGB -> Normalize
        img = ((img[:, :, ::-1] / 255) - 0.5) * 2
        images.append(img)
    
    num_cam = len(images)

    # masks
    mask_dir = os.path.join(root, "mask")
    mask_paths = sorted(glob.glob(f"{mask_dir}/*"))
    
    for i, mask_path in enumerate(mask_paths):
        mask = cv2.imread(mask_path)
        assert mask.shape[:2] == img_sizes[
            i], "Mask image imcompatible with RGB"

        # preprocess: BGR -> Gray -> Mask -> Tensor
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 127
        object_masks.append(mask)

    # SMPL
    shape = np.load(os.path.join(root, "mean_shape.npy"))
    pose_paths = sorted(glob.glob(f"{os.path.join(root, 'blazerlong_slack_trial2')}/*")) 
    trans = np.load(os.path.join(root, 'normalize_trans.npy'))

    # cameras
    cameras = dict(np.load(os.path.join(root, "cameras.npz")))

    Ps, Cs = [], []

    for i in range(num_cam):
        P = cameras[f"cam_{i}"].astype(np.float32)
        Ps.append(P)

        C = -np.linalg.solve(P[:3, :3], P[:3, 3])
        Cs.append(C)
    
    num_cam = len(pose_paths)
    pixel_per_batch = 4096
    os.makedirs("test_rendering_animate", exist_ok=True)
    for idx in range(0, num_cam):
        print("current frame:", idx)
        img_size = img_sizes[idx]
        total_pixels = np.prod(img_size)
        uv = np.mgrid[:img_size[0], :img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)
        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = 1
        smpl_params[1:4] = torch.from_numpy(trans[idx]).float()
        poses = np.load(pose_paths[idx])['pose']
        smpl_params[4:76] = torch.from_numpy(poses).float()
        smpl_params[76:] = torch.from_numpy(shape).float()
        num_batches = (total_pixels + pixel_per_batch -
                       1) // pixel_per_batch
        inputs = {
            "object_mask": object_masks[idx].reshape(-1).astype(bool),
            "uv": uv.reshape(-1, 2).astype(np.float32),
            "P": Ps[idx],
            "C": Cs[idx],
            "smpl_params": smpl_params
        }
        targets = {
            "rgb": images[idx].reshape(-1, 3).astype(np.float32),
            "img_size": img_sizes[idx]
        }
        results = []

        for b in trange(num_batches):
            indices = list(range(b * pixel_per_batch,
                                min((b + 1) * pixel_per_batch, total_pixels)))
            batch_inputs = {"object_mask": torch.tensor(inputs["object_mask"][indices][None]).cuda(),
                            "uv": torch.tensor(inputs["uv"][indices][None]).cuda(),
                            "P": torch.tensor(inputs["P"][None]).cuda(),
                            "C": torch.tensor(inputs["C"][None]).cuda(),
                            "smpl_params": inputs["smpl_params"][None].cuda()}

            with torch.no_grad():
                model_outputs = model(batch_inputs)

            batch_targets = {"rgb": torch.tensor(targets["rgb"][indices][None]).detach().clone(),
                            "img_size": targets["img_size"]}
            results.append({'rgb_values':model_outputs["rgb_values"].detach().clone(), 
                            **batch_targets})
        img_size = results[0]["img_size"]
        rgb_pred = torch.cat([result["rgb_values"] for result in results], dim=0)
        rgb_pred = (rgb_pred.reshape(*img_size, -1) + 1) / 2

        rgb_gt = torch.cat([result["rgb"] for result in results], dim=1).squeeze(0).cuda()
        rgb_gt = (rgb_gt.reshape(*img_size, -1) + 1) / 2

        rgb = torch.cat([rgb_gt, rgb_pred], dim=0).cpu().numpy()
        rgb = (rgb * 255).astype(np.uint8)

        cv2.imwrite(f"test_rendering_animate/{idx:04d}.png", rgb[:, :, ::-1])
            
if __name__ == '__main__':
    main()