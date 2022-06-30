import os


gender = 'MALE'

os.system('romp --mode=video --calc_smpl --render_mesh -i=Easy_on_me_720p_cut.mp4 -o=/home/chen/disk2/Youtube_Videos/ROMP/results.mp4 --save_video -t -sc=5 --smpl_path=/home/chen/romp/SMPL_%s.pth' % gender)