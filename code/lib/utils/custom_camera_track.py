import open3d as o3d
import numpy as np
def cameracenter_from_translation(R, t):
    t = t.reshape(-1, 3, 1)
    R = R.reshape(-1, 3, 3)
    C = -R.transpose(0, 2, 1) @ t
    return C.squeeze()

def cameras_lineset(Rs, ts, size=10, color=(0.5, 0.5, 0)):
    points = []
    lines = []
    colors = []

    for R, t in zip(Rs, ts):
        C0 = cameracenter_from_translation(R, t).reshape(3, 1)
        cam_points = []
        cam_points.append(C0)
        cam_points.append(
            C0 + R.T @ np.array([-size, -size, 3.0 * size]).reshape(3, 1)
        )
        cam_points.append(
            C0 + R.T @ np.array([+size, -size, 3.0 * size]).reshape(3, 1)
        )
        cam_points.append(
            C0 + R.T @ np.array([+size, +size, 3.0 * size]).reshape(3, 1)
        )
        cam_points.append(
            C0 + R.T @ np.array([-size, +size, 3.0 * size]).reshape(3, 1)
        )
        cam_points = np.concatenate(
            [pt.reshape(1, 3) for pt in cam_points], axis=0
        )
        cam_lines = np.array(
            [(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (2, 3), (3, 4), (4, 1)]
        )

        points.append(cam_points)
        lines.append(len(lines) * 5 + cam_lines)
        colors.extend([color for _ in range(8)])

    points = np.concatenate(points, axis=0)
    lines = np.concatenate(lines, axis=0)
    ls = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    ls.colors = o3d.utility.Vector3dVector(np.array(colors, dtype=np.float64))
    return ls

if __name__ == "__main__":
    cam = np.load('/home/chen/RGB-PINA/data/downtown_walkDownhill_00/cameras.npy') 

    src_Rs = cam[:1, :3, :3]
    src_ts = cam[:1, :3, 3]
    cam_size = 1
    o3d.visualization.draw_geometries(
        [
            cameras_lineset(src_Rs, src_ts, cam_size, (0.0, 0.85, 0.5)),
        ]
    )