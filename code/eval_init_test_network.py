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
    dataset_path = '/home/chen/disk2/AMASS/Initialization_Avatar/split'

    file_list = sorted(glob.glob(os.path.join(dataset_path, '*.pkl')))
    file_list = file_list[::100]

    smpl_params = torch.zeros([1, 72]).float().cuda()
    smpl_params[:, 0] = 1
    pose = torch.tensor(pkl.load(open(file_list[0], 'rb'))['poses']).float().unsqueeze(0).cuda() 
    # root = '/home/chen/snarf_idr_cg_1/data/cape'
    smpl_params[:, 4:67] = torch.tensor(pose[:, 3:66]).float()
    # pose = torch.zeros_like(pose).cuda()
    cond = {'smpl': smpl_params[:, 3:]/np.pi}

    sdf_init = SDF_Init(opt)# .load_from_checkpoint(checkpoint, opt=opt)
    model_state = torch.load('/home/chen/snarf_idr_cg_1/code/outputs/PosePrior/985.pth')
    implicit_network = sdf_init.model.implicit_network
    implicit_network.load_state_dict(model_state['model_state_dict'])
    implicit_network.eval()
    implicit_network = implicit_network.cuda()

    with torch.no_grad():
        mesh = mesh_from_implicit_func(
            func=lambda x, cond=cond: implicit_network(
                torch.from_numpy(x).float().cuda(), cond)[0, :, 0].cpu(),
            bbox=np.asarray([(-1, -1, -1), (1, 1, 1)]),
            coarse_bbox=True,
            resolution=(128, 128, 128))
    mesh.export("sample.ply")


if __name__ == '__main__':
    main()
