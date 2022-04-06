from lib.model.idr import IDR
from lib.utils.mesh import mesh_from_implicit_func
import glob
import hydra
import numpy as np
import torch
import pytorch_lightning as pl
from lib.model.smpl import SMPLServer
import os

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    smpl_server = SMPLServer(gender='male')
    # root = os.path.join("../data", opt.data_dir)
    # root = hydra.utils.to_absolute_path(root)
    root = '../data/buff'
    pose = torch.tensor(np.load(hydra.utils.to_absolute_path(os.path.join(root, 'poses.npy')))[0]).float().unsqueeze(0).cuda() 
    # pose = np.load('/home/chen/disk2/cape_release/full_cape_SCANimate_test/03375_blazerlong_lean_trial2/blazerlong_lean_trial2.000085.npz')['pose']
    # pose = torch.tensor(pose).float().unsqueeze(0).cuda()
    cond = {'smpl': pose[:, 3:]/np.pi}
    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    print("Checkpoint", checkpoint)
    betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.train.data_dir, 'mean_shape.npy')
    model = IDR.load_from_checkpoint(checkpoint, opt=opt, betas_path=betas_path)
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        mesh = mesh_from_implicit_func(
            func=lambda x, cond=cond: model.model.implicit_network(
                torch.from_numpy(x).float().cuda(), cond)[0, :, 0].cpu(),
            bbox=np.asarray([(-1, -1, -1), (1, 1, 1)]),
            coarse_bbox=True,
            resolution=(256, 256, 256))
    mesh.export("sample_cano.ply")


if __name__ == '__main__':
    main()
