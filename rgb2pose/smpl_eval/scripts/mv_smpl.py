import numpy as np
import os
import glob
from tqdm import trange
import sys
sys.path.append(os.getcwd())

from smpl_eval.utils.evaluator import SMPL_Evaluator


from smplx import SMPL

SMPL_PATH = "/media/yifei/c7996358-7c27-4915-b495-fe61422be573/models/smpl"
smpl_neutral_model = SMPL(model_path=SMPL_PATH, gender="neutral")

SMPL_JOINTS_NUM = 24
SCALER = 1000

GT_PATH = "/media/yifei/InteractionData2/Hi4D"
PRED_PATH = "/media/yifei/InteractionData2" 


def main(args):

    for pair in args.pairs:

        if len(args.actions) == 0:
            action_list = sorted(glob.glob(os.path.join(GT_PATH, pair, '*')))
            actions = [f.split('/')[-1] for f in action_list if pair[-2:] == f.split('/')[-1][-2:]]
        else:
            actions = args.actions

        for action in actions:
            print("pair: {} action: {}".format(pair, action))

            gt_root = os.path.join(GT_PATH, pair, action)
         
            total_count, total_misscount, total_fpcount = 0, 0, 0
            MPJPE, MVE, PAMPJPE, PAMVE  = [], [], [], []
            NUM_DEPTH, CORRECT_DEPTH = 0, 0
            CD = []

            evaluator = SMPL_Evaluator(gt_root)
            pred_root = os.path.join(PRED_PATH, pair, action, "results/mv_smpl", args.exp_name, f"{args.num_views}_views", "smpl_npz")

            for i in trange(evaluator.start, evaluator.end+1):
                
                # load data
                gt_verts, gt_joints_3d, pred_vertices, pred_joints_3d = evaluator.load_data_mv(i, pred_root, args.exp_name)

                if None in pred_joints_3d or None in pred_vertices:
                    total_misscount += gt_verts.shape[0]
                    continue

                # matching
                miss_flag, falsePositive_count, gt3d_verts, pred_verts, gt3d_joints, pred_joints = evaluator.matching_mv(gt_verts, gt_joints_3d, pred_vertices, pred_joints_3d)

                # evaluation
                errors_joints, errors_verts, errors_procrustes, errors_procrustes_verts,  correct_depth, depth_num,  contact_error = evaluator.compute_metrics_mv(i, gt3d_verts, pred_verts, gt3d_joints, pred_joints, miss_flag)

                total_count += len(pred_joints)
                total_misscount += sum(miss_flag)
                total_fpcount += falsePositive_count

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

            np.savez(os.path.join(pred_root, "eval_new.npz"), 
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
    parser.add_argument('--pairs', required=True, nargs='+' ,help='pairs to process')
    parser.add_argument('--actions', nargs='+', type=str, default=[])
    parser.add_argument('--exp_name', type=str, default='romp')
    parser.add_argument('--num_views', type=int, default=8)
    main(parser.parse_args())