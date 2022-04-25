from lib.model.volsdf_wo_snarf import VolSDF
from lib.utils.mesh import mesh_from_implicit_func
import glob
import hydra
import numpy as np
import torch
import pytorch_lightning as pl


@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)

    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    print("Checkpoint", checkpoint)
    model = VolSDF.load_from_checkpoint(checkpoint, opt=opt)
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        mesh = mesh_from_implicit_func(
            func=lambda x: model.model.implicit_network(
                torch.from_numpy(x).float().cuda())[:, 0].cpu(),
            bbox=np.asarray([(-1, -1, -1), (1, 1, 1)]),
            coarse_bbox=True,
            resolution=(256, 256, 256))
    mesh.export("sample.ply")


if __name__ == '__main__':
    main()
