from lib.model.sdf_init import SDF_Init
from lib.utils.mesh import mesh_from_implicit_func
import glob
import hydra
import numpy as np
import torch
import pytorch_lightning as pl
from lib.model.smpl import SMPLServer
import os
import pickle as pkl

@hydra.main(config_path="confs", config_name="base_init")
def main(opt):
    pl.seed_everything(42)
    smpl_server = SMPLServer(gender='male')
    # root = os.path.join("../data", opt.data_dir)
    # root = hydra.utils.to_absolute_path(root)

    dataset_path = '/home/chen/disk2/AMASS/Initialization_Avatar/split'
    

    file_list = sorted(glob.glob(os.path.join(dataset_path, '*.pkl')))
    file_list = file_list[::100]

    smpl_params = torch.zeros([1, 72]).float().cuda()
    smpl_params[:, 0] = 1
    pose = torch.tensor(pkl.load(open(file_list[0], 'rb'))['poses']).float().unsqueeze(0).cuda() 
    # root = '/home/chen/snarf_idr_cg_1/data/cape'
    smpl_params[:, 3:66] = torch.tensor(pose[:, 3:66]).float()
    # pose = torch.zeros_like(pose).cuda()
    cond = {'smpl': smpl_params[:, 3:]/np.pi}
    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[0]
    print("Checkpoint", checkpoint)

    model = SDF_Init.load_from_checkpoint(checkpoint, opt=opt)
    model.eval()
    model = model.cuda()

    with torch.no_grad():
        mesh = mesh_from_implicit_func(
            func=lambda x, cond=cond: model.model.implicit_network(
                torch.from_numpy(x).float().cuda(), cond)[0, :, 0].cpu(),
            bbox=np.asarray([(-1, -1, -1), (1, 1, 1)]),
            coarse_bbox=True,
            resolution=(128, 128, 128))
    mesh.export("sample.ply")


if __name__ == '__main__':
    main()
