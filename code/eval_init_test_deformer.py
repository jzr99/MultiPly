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
import trimesh
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
    smpl_params[:, 3:66] = torch.tensor(pose[:, 3:66]).float()
    # pose = torch.zeros_like(pose).cuda()
    cond = {'smpl': smpl_params[:, 3:]/np.pi}

    sdf_init = SDF_Init(opt)# .load_from_checkpoint(checkpoint, opt=opt)
    model_state = torch.load('/home/chen/snarf_idr_cg_1/code/outputs/smpl_init.pth')
    deformer = sdf_init.model.deformer
    deformer.load_state_dict(model_state['deformer_state_dict'])
    deformer.eval()
    deformer = deformer.cuda()
    smpl_server = sdf_init.model.smpl_server

    buff_mesh = trimesh.load('/home/chen/disk2/cape_release/raw_scans_textured/00096/shortlong_hips/shortlong_hips.000001.ply')
    buff_mesh.apply_scale(1/1000)
    buff_mesh = trimesh.graph.split(buff_mesh, only_watertight=False)[0]
    
    smpl_data = np.load('/home/chen/disk2/cape_release/sequences/00096/shortlong_hips/shortlong_hips.000001.npz')
    smpl_params = torch.zeros(1, 86).float().cuda()
    smpl_params[:, 0] = 1
    # smpl_params[:, 1: 4] = torch.tensor(smpl_data['transl']).float().cuda()
    smpl_params[:, 4: 76] = torch.tensor(smpl_data['pose']).float().cuda()
    buff_mesh.vertices -= smpl_data['transl']
    smpl_tfs = smpl_server(smpl_params)['smpl_tfs']
    with torch.no_grad():
        points_c = deformer.backward_skinning(torch.tensor(buff_mesh.vertices).unsqueeze(0).float().cuda(), None, smpl_tfs)
    mesh = trimesh.Trimesh(points_c.detach().cpu().numpy().squeeze(0))
    mesh.export("canonical_points_nn.ply")


if __name__ == '__main__':
    main()
