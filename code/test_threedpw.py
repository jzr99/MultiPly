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
    skip_step = 2
    betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.train.data_dir, 'mean_shape.npy')
    idr_model = IDR.load_from_checkpoint(checkpoint, opt=opt, betas_path=betas_path)
    model = idr_model.model
    model.eval()
    model = model.cuda()

    img_sizes = []
    images = []
    object_masks = []

    root = os.path.join("../data", opt.dataset.train.data_dir)
    root = hydra.utils.to_absolute_path(root)
    # images
    img_dir = os.path.join(root, "image")
    img_paths = sorted(glob.glob(f"{img_dir}/*"))

    for img_path in img_paths[::skip_step]:
        img = cv2.imread(img_path)
        img_size = img.shape[:2]
        img_sizes.append(img_size)

        # preprocess: BGR -> RGB -> Normalize
        img = ((img[:, :, ::-1] / 255) - 0.5) * 2
        images.append(img)
    
    num_imgs = len(images)

    # masks
    mask_dir = os.path.join(root, "mask")
    mask_paths = sorted(glob.glob(f"{mask_dir}/*"))
    
    for i, mask_path in enumerate(mask_paths[::skip_step]):
        mask = cv2.imread(mask_path)
        assert mask.shape[:2] == img_sizes[
            i], "Mask image imcompatible with RGB"

        # preprocess: BGR -> Gray -> Mask -> Tensor
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) > 127
        object_masks.append(mask)

    # SMPL
    shape = np.load(os.path.join(root, "mean_shape.npy"))
    poses = np.load(os.path.join(root, 'poses.npy'))[::skip_step]
    trans = np.load(os.path.join(root, 'normalize_trans.npy'))[::skip_step]

    # cameras
    cameras = np.load(os.path.join(root, "cameras.npy"))[::skip_step]

    Ps, Cs = [], []
    for i in range(cameras.shape[0]):
        P = cameras[i].astype(np.float32)
        Ps.append(P)

        C = -np.linalg.solve(P[:3, :3], P[:3, 3])
        Cs.append(C)


    pixel_per_batch = 6000
    os.makedirs("test_rendering", exist_ok=True)
    for idx in range(200, len(images)):
        print("current frame:", idx)
        img_size = img_sizes[idx]
        total_pixels = np.prod(img_size)
        uv = np.mgrid[:img_size[0], :img_size[1]].astype(np.int32)
        uv = np.flip(uv, axis=0).copy().transpose(1, 2, 0).astype(np.float32)
        smpl_params = torch.zeros([86]).float()
        smpl_params[0] = 1
        smpl_params[1:4] = torch.from_numpy(trans[idx]).float()
        smpl_params[4:76] = torch.from_numpy(poses[idx]).float()
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
                            "smpl_params": inputs["smpl_params"][None].cuda(),
                            "smpl_pose": inputs['smpl_params'][None][:, 4:76].cuda(),
                            "smpl_shape": inputs['smpl_params'][None][:, 76:].cuda(),
                            "smpl_trans": inputs['smpl_params'][None][:, 1:4].cuda()}

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

        cv2.imwrite(f"test_rendering/{idx:04d}.png", rgb[:, :, ::-1])
            
if __name__ == '__main__':
    main()