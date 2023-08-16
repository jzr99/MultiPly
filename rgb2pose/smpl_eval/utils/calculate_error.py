import numpy as np

SMPL_JOINTS = 24

def compute_similarity_transform(S1, S2, num_joints, verts=None):
    '''
    Computes a similarity transform (sR, t) that takes
    a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
    where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
    i.e. solves the orthogonal Procrutes problem.
    '''
    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        if verts is not None:
            verts = verts.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # Use only body joints for procrustes
    S1_p = S1[:, :num_joints]
    S2_p = S2[:, :num_joints]
    # 1. Remove mean.
    mu1 = S1_p.mean(axis=1, keepdims=True)
    mu2 = S2_p.mean(axis=1, keepdims=True)
    X1 = S1_p - mu1
    X2 = S2_p - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1 ** 2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Error:
    S1_hat = scale * R.dot(S1) + t

    if verts is not None:
        verts_hat = scale * R.dot(verts) + t
        if transposed:
            verts_hat = verts_hat.T

    if transposed:
        S1_hat = S1_hat.T

    procrustes_params = {'scale': scale,
                         'R': R,
                         'trans': t}

    if verts is not None:
        return S1_hat, verts_hat, procrustes_params
    else:
        return S1_hat, procrustes_params


def align_by_pelvis(joints, verts=None):

    left_id = 1
    right_id = 2

    pelvis = (joints[left_id, :] + joints[right_id, :]) / 2.0
    if verts is not None:
        return verts - np.expand_dims(pelvis, axis=0)
    else:
        return joints - np.expand_dims(pelvis, axis=0)

def compute_errors_joints_verts(gt_verts, pred_verts, gt_joints,
                                pred_joints, miss=None):

    # num_joints = gt_joints[0].shape[0]
    errors, errors_verts, errors_procrustes, errors_procrustes_verts = [], [], [], []

    for i, (gt3d, pred) in enumerate(zip(gt_joints, pred_joints)):
        # Get corresponding ground truth and predicted 3d joints and verts
        if miss[i] == 1:
            errors.append(0)
            errors_verts.append(0)
            continue
        gt3d = gt3d.reshape(-1, 3)
        gt3d_verts = gt_verts[i].reshape(-1, 3)
        pred3d_verts = pred_verts[i].reshape(-1, 3)

        gt3d_verts = align_by_pelvis(gt3d, gt3d_verts)
        pred3d_verts = align_by_pelvis(pred, pred3d_verts)
        gt3d = align_by_pelvis(gt3d)
        pred3d = align_by_pelvis(pred)

        # import ipdb; ipdb.set_trace()
        # Calculate joints and verts pelvis align error
        joint_error = np.sqrt(np.sum((gt3d - pred3d) ** 2, axis=1))
        verts_error = np.sqrt(np.sum((gt3d_verts - pred3d_verts) ** 2, axis=1))
        errors.append(np.mean(joint_error))
        errors_verts.append(np.mean(verts_error))


        # Get procrustes align error. # Not used anymore
        pred3d_sym, pred3d_verts_sym, procrustesParam = compute_similarity_transform(pred3d, gt3d, SMPL_JOINTS, pred3d_verts)

        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        pa_verts_error = np.sqrt(np.sum((gt3d_verts - pred3d_verts_sym) ** 2, axis=1))
        errors_procrustes.append(np.mean(pa_error))
        errors_procrustes_verts.append(np.mean(pa_verts_error))

    return errors, errors_verts, errors_procrustes, errors_procrustes_verts
    
def compute_errors_joints_verts_wo_align(gt_verts, pred_verts, gt_joints,
                                pred_joints, miss=None):

    # num_joints = gt_joints[0].shape[0]
    errors, errors_verts, errors_procrustes, errors_procrustes_verts = [], [], [], []

    for i, (gt3d, pred) in enumerate(zip(gt_joints, pred_joints)):
        # Get corresponding ground truth and predicted 3d joints and verts
        if miss[i] == 1:
            errors.append(0)
            errors_verts.append(0)
            continue
        gt3d = gt3d.reshape(-1, 3)
        gt3d_verts = gt_verts[i].reshape(-1, 3)
        pred3d_verts = pred_verts[i].reshape(-1, 3)

        # Calculate joints and verts pelvis align error
        joint_error = np.sqrt(np.sum((gt3d - pred) ** 2, axis=1))
        verts_error = np.sqrt(np.sum((gt3d_verts - pred3d_verts) ** 2, axis=1))
        errors.append(np.mean(joint_error))
        errors_verts.append(np.mean(verts_error))


        # Get procrustes align error. # Not used anymore
        pred3d_sym, pred3d_verts_sym, procrustesParam = compute_similarity_transform(pred, gt3d, SMPL_JOINTS, pred3d_verts)

        pa_error = np.sqrt(np.sum((gt3d - pred3d_sym) ** 2, axis=1))
        pa_verts_error = np.sqrt(np.sum((gt3d_verts - pred3d_verts_sym) ** 2, axis=1))
        errors_procrustes.append(np.mean(pa_error))
        errors_procrustes_verts.append(np.mean(pa_verts_error))

    return errors, errors_verts, errors_procrustes, errors_procrustes_verts

