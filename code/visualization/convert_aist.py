import numpy as np
import glob
import os

seq = 'gLO_sBM_cAll_d14_mLO1_ch05'
npz_file_paths = sorted(glob.glob(f'/home/chen/SCANimate/data/test/{seq}/seqs/*.npz'))
save_dir = f'/home/chen/RGB-PINA/animation_assets/{seq}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
output_pose = []
for npz_file_path in npz_file_paths:
    npz_file = dict(np.load(npz_file_path))
    output_pose.append(npz_file['pose'])
    np.save(os.path.join(save_dir, 'opt_poses.npy'), np.array(output_pose))