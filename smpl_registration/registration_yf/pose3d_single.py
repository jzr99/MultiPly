import os
import numpy as np

from tqdm import tqdm

import logging
from pathlib import Path
list_roof_camera_ids = [96, 97, 100, 101, 104]

def triangulate(projection_matrices, keypoints, reg=1e-3):
    """Triangulate with oridnary least square

    For a overdetermined linear system, the method of OLS is
        adopted to find an approximate solution.

    The solution of OLS satisfies following constraint:
        (A^TA) X = A^T b

    Args:
        projection_matrices (N, 3, 4)
        keypoints (N, J, 3)

    Return:
        keypoints_3d (J, 3)
    """
    J = keypoints.shape[1]
    ATA = np.zeros((J, 3, 4))
    ATA[:] = np.eye(3, 4) * reg
    for (P, keypoints_2d) in zip(projection_matrices, keypoints):
        x, y, w = keypoints_2d.T
        Ai = np.stack([x[:, None] * P[2] - P[0], y[:, None] * P[2] - P[1]],
                      axis=1)
        ATA += Ai.transpose(0, 2, 1)[:, :-1, :] @ Ai * w[:, None, None]

    keypoints_3d = np.ones((J, 4))
    keypoints_3d[:, :3] = np.linalg.solve(ATA[:, :, :-1], -ATA[:, :, -1])

    # validate triangulation
    reproj_errs, weights = [], []
    for (P, keypoints_2d) in zip(projection_matrices, keypoints):
        keypoints_3d_project = np.einsum("ij,kj->ik", keypoints_3d, P)
        keypoints_3d_project /= keypoints_3d_project[:, 2, None]

        err = ((keypoints_3d_project - keypoints_2d)[:, :2] ** 2).sum(-1)
        err = np.sqrt(err)

        reproj_errs.append(err)
        weights.append(keypoints_2d[:, 2])

    reproj_errs = np.stack(reproj_errs, axis=0)
    weights = np.stack(weights, axis=0)
    avg_loss = (reproj_errs * weights).sum(axis=0) / weights.sum(axis=0)
    avg_loss = avg_loss.mean()
    return keypoints_3d, avg_loss

def main(args):
    root = Path(args.capture_root)

    camera_path = root / "cameras" / "rgb_cameras.npz"
    keypoint_path = root / "pose2d" / "keypoints.npz"

    cameras = dict(np.load(camera_path))
    keypoints_2d = dict(np.load(keypoint_path))

    projection_matrices, keypoints_matrices = [], []

    for i in range(len(cameras["ids"])):
        cam_id = cameras["ids"][i]
        if cam_id in list_roof_camera_ids:
            continue
        keypoints = keypoints_2d[str(cam_id)]

        intrinsic = cameras["intrinsics"][i]
        extrinsic = cameras["extrinsics"][i]
        projection_matrix = intrinsic @ extrinsic
        projection_matrices.append(projection_matrix)
        keypoints_matrices.append(keypoints)

    projection_matrices = np.stack(projection_matrices, axis=0)
    keypoints_matrices = np.concatenate(keypoints_matrices, axis=1)

    keypoints_3d_all, reproj_losses = [], []
    with tqdm(keypoints_matrices) as pbar:
        for keypoints_matrix in pbar:
            keypoints_3d, reproj_loss = triangulate(
                projection_matrices, keypoints_matrix)
            keypoints_3d_all.append(keypoints_3d)
            reproj_losses.append(reproj_loss)
            pbar.set_postfix_str(f"reprojection err: {reproj_loss:.2f} pixels")
    reproj_losses = np.asarray(reproj_losses)
    keypoints_3d_all = np.asarray(keypoints_3d_all)

    print(f"Reproj err: {reproj_losses.mean():.2f} (avg), {reproj_losses.max():.2f} (max) {reproj_losses.min():.2f} (min)")

    out_dir = root / "pose-estimation"
    os.makedirs(out_dir, exist_ok=True)
    output_kpts = os.path.join(out_dir, 'keypoints-3d.npy')
    np.save(output_kpts, keypoints_3d_all)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture_root', required=True)
    main(parser.parse_args())