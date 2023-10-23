import os
import joblib
import numpy as np

from smpl_eval.utils.projection import project_to_2d, word2cam
from smpl_eval.utils.matching import match_2d_greedy, match_3d_greedy, get_matching_dict
from smpl_eval.utils.calculate_error import compute_errors_joints_verts, compute_errors_joints_verts_wo_align
from smpl_eval.utils.estimate_trans import estimate_translation, depth_relation
from smpl_eval.utils.contact import contact_dist

SMPL_JOINTS_NUM = 24

class SMPL_Evaluator():
    def __init__(self, gt_root, cam=None):
        self.gt_root = gt_root

        # get the meta information
        meta = dict(np.load(os.path.join(self.gt_root, "meta.npz")))
        self.num_persons = meta["num_persons"]

        self.start = int(meta["start"])
        self.end = int(meta["end"])
        self.contact_frames = meta["contact_ids"].tolist()

         # get camera parameters
        if cam is None:
            self.cam = meta["mono_cam"]
        else:
            self.cam = cam
        print("cam: ", self.cam)
        cameras = dict(np.load(os.path.join(self.gt_root, "cameras", "rgb_cameras.npz")))
        c = int(np.where(cameras['ids'] == self.cam)[0])
        self.gt_cam_intrinsics = cameras['intrinsics'][c]
        self.gt_cam_extrinsics = cameras['extrinsics'][c]
        self.P = self.gt_cam_intrinsics @ self.gt_cam_extrinsics


    def load_data_mono(self, i, pred_root, exp_name=None):
        # load ground truth and prediction data 
        gt = np.load(os.path.join(self.gt_root, "smpl", f"{i:06d}.npz"))         
        gt_joints_2d = project_to_2d(gt["joints_3d"], self.P)
        # project ground truth from world to camera coordinate system
        gt_verts = word2cam(gt["verts"], self.gt_cam_extrinsics)
        gt_joints_3d = word2cam(gt["joints_3d"], self.gt_cam_extrinsics)
        self.gt_contact_label = gt["contact"]
        pred_vertices, pred_joints_2d, pred_joints_3d, pred_cam_trans = None, None, None, None

        if exp_name == "bev":
            pred_path = os.path.join(pred_root, f"{i:06d}__2_0.08.npz")
            pred = np.load(pred_path, allow_pickle=True)['results'][()]
            pred_joints_2d = pred["pj2d_org"][:,:SMPL_JOINTS_NUM]
            pred_joints_3d = pred["joints"][:,:SMPL_JOINTS_NUM]
            pred_vertices = pred["verts"]
            pred_cam_trans = pred["cam_trans"]
        elif exp_name == "romp":
            pred_path = os.path.join(pred_root, f"{i:06d}.npz")
            pred = np.load(pred_path, allow_pickle=True)['results'][()]
            pred_joints_2d = pred["pj2d_org"][:,:SMPL_JOINTS_NUM]
            pred_joints_3d = pred["joints"][:,:SMPL_JOINTS_NUM]
            pred_vertices = pred["verts"]
            pred_cam_trans = pred["cam_trans"]
        elif exp_name == 'pare':
            pred_path = os.path.join(pred_root, "pare_results", f"{i:06d}.pkl")
            pred = joblib.load(pred_path)
            pred_joints_2d = pred["smpl_joints2d"][:,:SMPL_JOINTS_NUM]
            pred_joints_3d = pred["smpl_joints3d"][:,:SMPL_JOINTS_NUM]
            pred_vertices = pred["smpl_vertices"]
            pred_cam_trans = pred["pred_cam_t"]
        
        elif exp_name == "v2a":
            pass

        else:
            Exception("Method not supported")

        return gt_verts, gt_joints_2d, gt_joints_3d, pred_vertices, pred_joints_2d, pred_joints_3d, pred_cam_trans
    
    def matching_mono(self, gt_verts, gt_joints_2d, gt_joints_3d, pred_vertices, pred_joints_2d, pred_joints_3d, pred_cam_trans):
        gt3d_verts, pred_verts, gt3d_joints, pred_joints, pred_trans = [], [], [], [], []

        # matching the ground truth and prediction joints
        matching = match_2d_greedy(pred_joints_2d, gt_joints_2d, iou_thresh=0.1)
        matchDict, falsePositive_count = get_matching_dict(matching)

        gtIdxs = np.arange(self.num_persons)
        miss_flag = []
    
        for gtIdx in gtIdxs:
            gt3d_verts.append(gt_verts[gtIdx])
            gt3d_joints.append(gt_joints_3d[gtIdx])
            if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                miss_flag.append(1)
                pred_verts.append([])
                pred_joints.append([])
                pred_trans.append([])
            else:
                miss_flag.append(0)
                pred_joints.append(pred_joints_3d[matchDict[str(gtIdx)]])
                pred_verts.append(pred_vertices[matchDict[str(gtIdx)]])
                cam_trans = estimate_translation(pred_joints_3d[[matchDict[str(gtIdx)]]], pred_joints_2d[[matchDict[str(gtIdx)]]], pred_cam_trans[[matchDict[str(gtIdx)]]], proj_mats=self.gt_cam_intrinsics)
                pred_trans.append(cam_trans.squeeze(0).numpy())
                # pred_trans.append(pred_cam_trans[matchDict[str(gtIdx)]])
                
        return miss_flag, falsePositive_count, gt3d_verts, pred_verts, gt3d_joints, pred_joints, pred_trans

    def compute_metrics_mono(self, i, gt3d_verts, pred_verts, gt3d_joints, pred_joints, pred_trans, miss_flag):

        errors_joints, errors_verts, errors_procrustes, errors_procrustes_verts = compute_errors_joints_verts(gt3d_verts, pred_verts, gt3d_joints, pred_joints, miss_flag)
        joint_correct_depth = 0
        joint_depth_num = 0
        # 3D Relation
        if not (1 in miss_flag):
            depth_num = 1
            gt_depth_dist = (gt3d_joints[0][0,2]-gt3d_joints[1][0,2])
            gt_depth_order = depth_relation(gt_depth_dist, 0.15)

            pred_depth_dist = (pred_joints[0][0,2]+pred_trans[0][2])-(pred_joints[1][0,2]+pred_trans[1][2])
            # pred_depth_dist = (pred_joints[0][0,2] - pred_joints[1][0,2])
            pred_depth_order = depth_relation(pred_depth_dist, 0.15)

            if gt_depth_order == pred_depth_order:
                correct_depth = 1
            else:
                correct_depth = 0

            for keypoint_id in range(len(gt3d_joints[0])):
                joint_depth_num = joint_depth_num + 1
                gt_depth_dist = (gt3d_joints[0][keypoint_id, 2] - gt3d_joints[1][keypoint_id, 2])
                gt_joint_depth_order = depth_relation(gt_depth_dist, 0.1)
                pred_depth_dist = (pred_joints[0][keypoint_id, 2] + pred_trans[0][2]) - (pred_joints[1][keypoint_id, 2] + pred_trans[1][2])
                # pred_joint_depth_dist = (pred_joints[0][0,2] - pred_joints[1][0,2])
                pred_joint_depth_order = depth_relation(pred_depth_dist, 0.15)
                if gt_joint_depth_order == pred_joint_depth_order:
                    joint_correct_depth = joint_correct_depth + 1

            # Contact distance
            contact_error = 0
            if i in self.contact_frames:   
                pred0 = pred_verts[0] + pred_trans[0]
                pred1 = pred_verts[1] + pred_trans[1]
                error_contact_dist = contact_dist(self.gt_contact_label, pred0, pred1)
                
                if len(error_contact_dist) > 0:
                    contact_error =np.mean(np.concatenate(error_contact_dist))
        else:
            depth_num = 0
            correct_depth = 0
            contact_error = 0
            

        return errors_joints, errors_verts, errors_procrustes, errors_procrustes_verts, correct_depth, depth_num, contact_error, joint_correct_depth, joint_depth_num
    
    def load_data_mv(self, i, pred_root, exp_name=None):
        # load ground truth and prediction data 
        gt = np.load(os.path.join(self.gt_root, "smpl", f"{i:06d}.npz"))         
        self.gt_contact_label = gt["contact"]

        if exp_name == "easymocap":
            pred =  np.load(os.path.join(pred_root, f"{i:06d}.npz"), allow_pickle=True)
        else:
            Exception("Method not supported")

        return gt["verts"], gt["joints_3d"], pred["vertices"], pred["joints"],
    
    def matching_mv(self, gt_verts, gt_joints_3d, pred_vertices, pred_joints_3d):
        gt3d_verts, pred_verts, gt3d_joints, pred_joints = [], [], [], []

        # matching the ground truth and prediction joints
        matching = match_3d_greedy(pred_joints_3d, gt_joints_3d)
        matchDict, falsePositive_count = get_matching_dict(matching)

        gtIdxs = np.arange(self.num_persons)
        miss_flag = []
    
        for gtIdx in gtIdxs:
            gt3d_verts.append(gt_verts[gtIdx])
            gt3d_joints.append(gt_joints_3d[gtIdx])
            if matchDict[str(gtIdx)] == 'miss' or matchDict[str(gtIdx)] == 'invalid':
                miss_flag.append(1)
                pred_verts.append([])
                pred_joints.append([])
            else:
                miss_flag.append(0)
                pred_joints.append(pred_joints_3d[matchDict[str(gtIdx)]][:SMPL_JOINTS_NUM])
                pred_verts.append(pred_vertices[matchDict[str(gtIdx)]])

        return miss_flag, falsePositive_count, gt3d_verts, pred_verts, gt3d_joints, pred_joints
    

    def compute_metrics_mv(self, i, gt3d_verts, pred_verts, gt3d_joints, pred_joints, miss_flag):

        errors_joints, errors_verts, errors_procrustes, errors_procrustes_verts = compute_errors_joints_verts_wo_align(gt3d_verts, pred_verts, gt3d_joints, pred_joints, miss_flag)

        if not (1 in miss_flag):
            depth_num = 1
            gt_depth_dist = (gt3d_joints[0][0,2]-gt3d_joints[1][0,2])
            gt_depth_order = depth_relation(gt_depth_dist, 0.1)

            pred_depth_dist = pred_joints[0][0,2]-pred_joints[1][0,2]
            pred_depth_order = depth_relation(pred_depth_dist, 0.1)

            if gt_depth_order == pred_depth_order:
                correct_depth = 1
            else:
                correct_depth = 0

            # Contact distance
            contact_error = 0
            if i in self.contact_frames:   
                pred0 = pred_verts[0] 
                pred1 = pred_verts[1] 
                error_contact_dist = contact_dist(self.gt_contact_label, pred0, pred1)
                
                if len(error_contact_dist) > 0:
                    contact_error =np.mean(np.concatenate(error_contact_dist))
        else:
            depth_num = 0
            correct_depth = 0
            contact_error = 0
            
            

        return errors_joints, errors_verts, errors_procrustes, errors_procrustes_verts, correct_depth, depth_num, contact_error
    

    

                               
