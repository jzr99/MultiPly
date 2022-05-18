from lib.model.volsdf import VolSDF
from lib.utils.mesh import mesh_from_implicit_func
import glob
import hydra
import numpy as np
import torch
import pytorch_lightning as pl
import os

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)

    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    print("Checkpoint", checkpoint)
    root = '../data/mocap_juan_waving'
    pose = torch.tensor(np.load(hydra.utils.to_absolute_path(os.path.join(root, 'poses.npy')))[0]).float().unsqueeze(0).cuda() 
    cond = {'smpl': pose[:, 3:]/np.pi}
    betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.train.data_dir, 'mean_shape.npy')
    model = VolSDF.load_from_checkpoint(checkpoint, opt=opt, betas_path=betas_path)
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        mesh = mesh_from_implicit_func(
            func=lambda x, cond=cond: model.model.implicit_network(
                torch.from_numpy(x).float().cuda(), cond)[0, :, 0].cpu(),
            bbox=np.asarray([(-2, -2, -2), (2, 2, 2)]),
            coarse_bbox=True,
            resolution=(128, 128, 128))
    mesh.export("sample.ply")


if __name__ == '__main__':
    main()
