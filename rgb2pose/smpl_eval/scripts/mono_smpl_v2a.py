import numpy as np
import os
import glob
from tqdm import trange
import sys
sys.path.append(os.getcwd())

from smpl_eval.utils.evaluator import SMPL_Evaluator


from smplx import SMPL

# SMPL_PATH = "/media/yifei/c7996358-7c27-4915-b495-fe61422be573/models/smpl"
# smpl_neutral_model = SMPL(model_path=SMPL_PATH, gender="neutral")

SMPL_JOINTS_NUM = 24
SCALER = 1000

GT_PATH = "/media/ubuntu/hdd/Hi4D_gt"
# GT_PATH = "/media/yifei/InteractionData2/Hi4D"
# PRED_PATH = "/media/yifei/InteractionData2" 

EXP_ROOT_path = "/media/ubuntu/hdd/V2A_output/"
EXP_path_list = [
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_temporal",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose_naive",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_temporal_SAM_auto_raw_openpose_smpl_surface_earlystop_certain",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_surface2_depth_samGT_personid",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_meshdepth",
]

def main(args):

    for pair in args.pairs:

        if len(args.actions) == 0:
            action_list = sorted(glob.glob(os.path.join(GT_PATH, pair, '*')))
            actions = [f.split('/')[-1] for f in action_list if pair[-2:] == f.split('/')[-1][-2:]]
        else:
            actions = args.actions

        for action in actions:
            print("pair: {} action: {}".format(pair, action))

            for PRED_PATH in EXP_path_list:
                gt_root = os.path.join(GT_PATH, pair, action)
                print("pred_path: {}".format(PRED_PATH))
                print("gt_root: {}".format(gt_root))
                total_count, total_misscount, total_fpcount = 0, 0, 0
                MPJPE, MVE, PAMPJPE, PAMVE  = [], [], [], []
                NUM_DEPTH, CORRECT_DEPTH = 0, 0
                CD = []
                assert len(args.cams)  == 1
                evaluator = SMPL_Evaluator(gt_root, cam=args.cams[0])
                # pred_root = os.path.join(PRED_PATH, pair, action, "results/mono_smpl", args.exp_name, str(evaluator.cam))
                pred_root = os.path.join(PRED_PATH, 'joint_opt_smpl_joint', 'all.npz')
                pred = np.load(pred_root, allow_pickle=True)

                for pred_idx, i in enumerate(trange(evaluator.start, evaluator.end+1)):
                    
                    # load data
                    gt_verts, gt_joints_2d, gt_joints_3d, pred_vertices, pred_joints_2d, pred_joints_3d, pred_cam_trans = evaluator.load_data_mono(i, pred_root, args.exp_name)
                    pred_vertices = pred['verts'][:, pred_idx]
                    pred_joints_2d = pred['pj2d_org'][:, pred_idx, :24] * 2
                    pred_joints_3d = pred['joints'][:, pred_idx, :24]
                    pred_cam_trans = pred['cam_trans'][:, pred_idx]
                    # import pdb;pdb.set_trace()
                    # matching
                    miss_flag, falsePositive_count, gt3d_verts, pred_verts, gt3d_joints, pred_joints, pred_trans = evaluator.matching_mono(gt_verts, gt_joints_2d, gt_joints_3d, pred_vertices, pred_joints_2d, pred_joints_3d, pred_cam_trans)

                    # evaluation
                    errors_joints, errors_verts, errors_procrustes, errors_procrustes_verts, correct_depth, depth_num, contact_error = evaluator.compute_metrics_mono(i, gt3d_verts, pred_verts, gt3d_joints, pred_joints, pred_trans, miss_flag)

                    total_count += len(pred_joints)
                    total_misscount += sum(miss_flag)
                    total_fpcount += falsePositive_count

                    # import pdb;pdb.set_trace()

                    MPJPE = sum([MPJPE,errors_joints],[])
                    MVE = sum([MVE,errors_verts],[])
                    PAMPJPE = sum([PAMPJPE,errors_procrustes],[])
                    PAMVE = sum([PAMVE,errors_procrustes_verts],[])
                    CD.append(contact_error)

                    NUM_DEPTH += depth_num
                    CORRECT_DEPTH += correct_depth
                        
                # filter out invalid frames
                MPJPE = [x for x in MPJPE if x != 0.0]
                MVE = [x for x in MVE if x != 0.0]
                PAMPJPE = [x for x in PAMPJPE if x != 0.0]
                PAMVE = [x for x in PAMVE if x != 0.0]
                CD = [x for x in CD if x != 0.0]
                print("MPJPE: {} MVE: {} PAMPJPE: {} PAMVE: {} CD: {} PCDR:  {}, {}".format(np.mean(MPJPE), np.mean(MVE), np.mean(PAMPJPE), np.mean(PAMVE), np.mean(CD), CORRECT_DEPTH, NUM_DEPTH))
                print("total_count: {} total_misscount: {} total_fpcount: {}".format(total_count, total_misscount, total_fpcount))
                np.savez(os.path.join(PRED_PATH, "eval_new.npz"), 
                        MPJPE = np.array(MPJPE), 
                        MVE = np.array(MVE),
                        PAMPJPE = np.array(PAMPJPE),
                        PAMVE = np.array(PAMVE),
                        PCDR= np.array([CORRECT_DEPTH, NUM_DEPTH]),
                        contact_dist = np.array(CD),
                        total_count = total_count,
                        total_fpcount = total_fpcount,
                        total_misscount = total_misscount)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pairs', required=True, nargs='+' ,help='pairs to process')
    # parser.add_argument('--actions', nargs='+', type=str, default=[])
    # parser.add_argument('--cams', nargs='+', type=int, default= [4]) 
    # parser.add_argument('--exp_name', type=str, default='v2a')
    parser.add_argument('--pairs',nargs='+' , type=str, default=["pair19"])
    parser.add_argument('--actions', nargs='+', type=str, default=["piggyback19"])
    parser.add_argument('--cams', nargs='+', type=int, default= [4])
    parser.add_argument('--exp_name', type=str, default='v2a')
    main(parser.parse_args())