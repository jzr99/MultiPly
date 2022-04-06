from lib.model.idr import IDR
from lib.utils.mesh import mesh_from_implicit_func
import glob
import hydra
import numpy as np
import torch
import pytorch_lightning as pl
# from lib.model.smpl import SMPLServer
import os

@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    
    # root = os.path.join("../data", opt.data_dir)
    # root = hydra.utils.to_absolute_path(root)
    root = '../data/buff'
    pose_index=0
    # poses = np.load(hydra.utils.to_absolute_path(os.path.join(root, f'opt_pose/smpl_pose_{pose_index}.npy')))
    betas_path = os.path.join(hydra.utils.to_absolute_path('..'), 'data', opt.dataset.train.data_dir, 'mean_shape.npy')
    betas = np.load(betas_path)
    pose = np.load(sorted(glob.glob("opt_pose/*.npy"))[-1]) 

    # pose = torch.tensor(poses[pose_index]).float().unsqueeze(0).cuda()
    # pose = np.load('/home/chen/disk2/cape_release/full_cape_SCANimate_test/03375_blazerlong_lean_trial2/blazerlong_lean_trial2.000085.npz')['pose']
    pose = torch.tensor(pose).float().unsqueeze(0).cuda()
    betas = torch.tensor(betas).float().unsqueeze(0).cuda()

    smpl_params = torch.zeros([1, 86]).float().cuda()
    smpl_params[:, 0] = 1
    smpl_params[:, 4:76] = pose[:]
    smpl_thetas = smpl_params[:, 4:76]
    cond = {'smpl': smpl_thetas[:, 3:]/np.pi}
    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    print("Checkpoint", checkpoint)

    model = IDR.load_from_checkpoint(checkpoint, opt=opt, betas_path=betas_path)
    model.eval()
    model = model.cuda()
    # use all bones for initialization during testing
    model.model.deformer.init_bones = np.arange(24)
    smpl_server = model.model.smpl_server
    scale, transl, thetas, betas = torch.split(smpl_params, [1, 3, 72, 10], dim=1)
    smpl_output = smpl_server.forward(scale, transl, thetas, betas)
    smpl_tfs = smpl_output['smpl_tfs']
    smpl_verts = smpl_output['smpl_verts']
    # import pdb
    # pdb.set_trace()
    with torch.no_grad():
        mesh = mesh_from_implicit_func(
            func=lambda x, cond=cond: model.model.sdf_func(
                torch.from_numpy(x).float().cuda(), cond, smpl_tfs, eval_mode=True)[0].cpu(),
            bbox=np.asarray([(-1, -1, -1), (1, 1, 1)]),
            coarse_bbox=True,
            resolution=(128, 128, 128))
    mesh.export("sample_posed.ply")


if __name__ == '__main__':
    main()
