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

GT_PATH = "/cluster/project/infk/hilliges/jiangze/ROMP/global_scratch/ROMP/ROMP/dataset/Hi4D/Hi4D_all/Hi4D"
# GT_PATH = "/media/yifei/InteractionData2/Hi4D"
# PRED_PATH = "/media/yifei/InteractionData2" 

EXP_ROOT_path = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/"
RAW_ROOT_path = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/preprocessing/raw_data/"
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
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_nodelay",
    # # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_silhouette"
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_nodelay_nodepth",
    # EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_loop",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_joint",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_loop_50_5",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_loop_100_10",
# RAW_ROOT_path + "pair16_jump16_4",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_loop",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_loop",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_depth_loop",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_delay_depth_loop",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_loop",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_loop",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_depth_loop",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth_loop",
# RAW_ROOT_path + "pair15_fight15_4_custom",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth_loop_align",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth10_loop_align",
# RAW_ROOT_path + "pair17_dance17_28",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_loop",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_loop",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_depth_loop",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_delay_depth_loop",
# EXP_ROOT_path + "Hi4D_pair18_basketball18_4_loop",
# EXP_ROOT_path + "Hi4D_pair18_basketball18_4_sam_loop",
# EXP_ROOT_path + "Hi4D_pair18_basketball18_4_sam_depth_loop",
# EXP_ROOT_path + "Hi4D_pair18_basketball18_4_sam_delay_depth_loop",
# RAW_ROOT_path + "pair18_basketball18_4",

# RAW_ROOT_path + "pair19_piggyback19_4",
# RAW_ROOT_path + "Hi4D_pair19_piggyback19_cam4_raw_openpose",
# RAW_ROOT_path + "Hi4D_pair19_piggyback19_cam4_raw_openpose_test",
# RAW_ROOT_path + "pair16_jump16_vitpose_4",

# RAW_ROOT_path + "pair19_piggyback19_custom_vitpose_4",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_vitpose",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_vitpose_trans",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_nodelay_vitpose_trans",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_nodelay_nodepth_vitpose",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_nodepth_vitpose",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_vitpose_trans_d005",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_vitpose_trans_d005_end_render",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_loop_depth_samGT_personid_triplane_meshprompt_nerfacc_pyrenderdepth_condpose_inter_vitpose_trans_d005_end_render_pose",
# RAW_ROOT_path + "pair16_jump16_vitpose_4",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_loop_align_vitpose",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_loop_align_vitpose",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_depth_loop_align_vitpose",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose",
# RAW_ROOT_path + "pair15_fight15_vitpose_4",
# RAW_ROOT_path + "pair16_jump16_vitpose_4",
# RAW_ROOT_path + "pair17_dance17_vitpose_28",
# RAW_ROOT_path + "pair18_basketball18_vitpose_4",
# EXP_ROOT_path + "Hi4D_pair18_basketball18_4_loop_vitpose",
# EXP_ROOT_path + "Hi4D_pair18_basketball18_4_sam_loop_vitpose",
# EXP_ROOT_path + "Hi4D_pair18_basketball18_4_sam_depth_loop_vitpose",
# EXP_ROOT_path + "Hi4D_pair18_basketball18_4_sam_delay_depth_loop_vitpose"
# RAW_ROOT_path + "pair19_piggyback19_vitpose_4",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_sam",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_sam_depth",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_sam_delay_depth",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_loop_align_vitpose",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_loop_align_vitpose",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_depth_loop_align_vitpose",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_depth_loop_align_vitpose_end",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end",

# EXP_ROOT_path + "Hi4D_pair15_fight15_4_loop_vitpose",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_loop_vitpose",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_depth_loop_vitpose",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_all",

# RAW_ROOT_path + "pair17_dance17_vitpose_28",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_loop_0pose_vitpose",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_loop_0pose_vitpose",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_depth_loop_0pose_vitpose",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_delay_depth_loop_0pose_vitpose",

# RAW_ROOT_path + "pair19_piggyback19_vitpose_4",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_2",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_sam_2",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_sam_delay_depth_2",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_sam_delay_depth_2_head_drender",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_2_noshare",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_sam_2_noshare",
# EXP_ROOT_path + "Hi4D_pair19_piggyback19_4_sam_delay_depth_2_noshare",

# RAW_ROOT_path + "pair15_fight15_vitpose_4",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_loop_vitpose_2",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_loop_vitpose_2",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_2",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_2_drender",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_2",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_loop_vitpose_2_noshare",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_loop_vitpose_2_noshare",
# EXP_ROOT_path + "Hi4D_pair15_fight15_4_sam_delay_depth_loop_vitpose_edge_2_noshare",

# RAW_ROOT_path + "pair16_jump16_vitpose_4",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_loop_align_vitpose",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_loop_align_vitpose",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2",
EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2_drender",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_loop_align_vitpose_noshare",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_loop_align_vitpose_noshare",
# EXP_ROOT_path + "Hi4D_pair16_jump16_4_sam_delay_depth_loop_align_vitpose_end_2_edge_noshare",

# RAW_ROOT_path + "pair17_dance17_vitpose_28",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_loop_0pose_vitpose_2",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_loop_0pose_vitpose_2",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_delay_depth_loop_0pose_vitpose_2",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_loop_0pose_vitpose_2_noshare",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_loop_0pose_vitpose_2_noshare",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_delay_depth_loop_0pose_vitpose_2_noshare",
# EXP_ROOT_path + "Hi4D_pair17_dance17_28_sam_delay_depth_loop_0pose_vitpose_2_drender",
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
                NUM_JOINT_DEPTH, CORRECT_JOINT_DEPTH = 0, 0
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
                    errors_joints, errors_verts, errors_procrustes, errors_procrustes_verts, correct_depth, depth_num, contact_error, joint_correct_depth, joint_depth_num = evaluator.compute_metrics_mono(i, gt3d_verts, pred_verts, gt3d_joints, pred_joints, pred_trans, miss_flag)

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

                    NUM_JOINT_DEPTH += joint_depth_num
                    CORRECT_JOINT_DEPTH += joint_correct_depth
                        
                # filter out invalid frames
                MPJPE = [x for x in MPJPE if x != 0.0]
                MVE = [x for x in MVE if x != 0.0]
                PAMPJPE = [x for x in PAMPJPE if x != 0.0]
                PAMVE = [x for x in PAMVE if x != 0.0]
                CD = [x for x in CD if x != 0.0]
                print("MPJPE: {} MVE: {} PAMPJPE: {} PAMVE: {} CD: {} PCDR:  {}, {}, JPCDR: {}, {}".format(np.mean(MPJPE), np.mean(MVE), np.mean(PAMPJPE), np.mean(PAMVE), np.mean(CD), CORRECT_DEPTH, NUM_DEPTH, CORRECT_JOINT_DEPTH, NUM_JOINT_DEPTH))
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
                        total_misscount = total_misscount,
                         NUM_JOINT_DEPTH = NUM_JOINT_DEPTH,
                         CORRECT_JOINT_DEPTH = CORRECT_JOINT_DEPTH)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument('--pairs', required=True, nargs='+' ,help='pairs to process')
    # parser.add_argument('--actions', nargs='+', type=str, default=[])
    # parser.add_argument('--cams', nargs='+', type=int, default= [4]) 
    # parser.add_argument('--exp_name', type=str, default='v2a')
    # parser.add_argument('--pairs',nargs='+' , type=str, default=["pair19"])
    # parser.add_argument('--actions', nargs='+', type=str, default=["piggyback19"])
    parser.add_argument('--pairs',nargs='+' , type=str, default=["pair16"])
    parser.add_argument('--actions', nargs='+', type=str, default=["jump16"])
    # parser.add_argument('--pairs',nargs='+' , type=str, default=["pair15"])
    # parser.add_argument('--actions', nargs='+', type=str, default=["fight15"])
    # parser.add_argument('--cams', nargs='+', type=int, default= [4])
    # parser.add_argument('--pairs',nargs='+' , type=str, default=["pair17"])
    # parser.add_argument('--actions', nargs='+', type=str, default=["dance17"])
    # parser.add_argument('--cams', nargs='+', type=int, default= [28])
    # parser.add_argument('--pairs',nargs='+' , type=str, default=["pair18"])
    # parser.add_argument('--actions', nargs='+', type=str, default=["basketball18"])
    parser.add_argument('--cams', nargs='+', type=int, default= [4])
    parser.add_argument('--exp_name', type=str, default='v2a')
    main(parser.parse_args())