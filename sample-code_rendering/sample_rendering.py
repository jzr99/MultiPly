import pyrender
import numpy as np
import cv2
import trimesh
import os

WIDTH = 1600
HEIGHT = 1200
FOCAL = 1600

def create_lookat_matrix(eye, at, up):
    forward = at - eye
    forward /= np.linalg.norm(forward)
    up = up / np.linalg.norm(up)
    right = np.cross(forward, up)
    up = np.cross(right, forward)

    lookat = np.eye(4)
    lookat[0, :3] = right     # x axis
    lookat[1, :3] = -up       # y axis (negate up b/c right handed)
    lookat[2, :3] = forward   # z axis
    lookat[:3, 3] = -lookat[:3, :3] @ eye
    return lookat
    
    
def get_camera_rig(num_cameras, axis=(0.5, 1.5, 0.5), C=(0, 0, 1), at=(0.5, 0.5, 0.5)):
    # convert to np.ndarray
    axis = np.array(axis) / np.linalg.norm(axis)
    C = np.array(C)
    at = np.array(at)

    extrinsics = []
    for i in range(num_cameras):
        angle = (1.0 * i / num_cameras) * 2 * np.pi
        R, _ = cv2.Rodrigues(axis * angle)

        eye = R @ C
        up = np.array((0, 1, 0))
        extrinsic = create_lookat_matrix(eye, at, up)
        extrinsics.append(extrinsic)
    return extrinsics

def get_cameras(num_cameras):
    cameras = []
    for C in [(0, 0, 2.1), (0, 0.5, 2.1), (0, 1, 2.1)]:
        for extrinsic in get_camera_rig(num_cameras, C=C):
            camera = pyrender.IntrinsicsCamera(fx=FOCAL,
                                               fy=FOCAL,
                                               cx=WIDTH / 2,
                                               cy=HEIGHT / 2)
            cameras.append((camera, extrinsic))
    return cameras


def render_image(cameras, meshes):
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])
    for mesh in meshes:
        mesh = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh)

    camera_nodes = []
    for (camera, extrinsic) in cameras:
        extrinsic = extrinsic.copy()
        extrinsic[[1, 2]] *= -1
        pose = np.linalg.inv(extrinsic)
        camera_node = scene.add(camera, pose=pose)
        camera_nodes.append(camera_node)

    color_imgs, depth_imgs = [], []
    renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)
    for camera_node in camera_nodes:
        scene.main_camera_node = camera_node
        color, depth = renderer.render(scene)
        color_imgs.append(cv2.cvtColor(color, cv2.COLOR_RGB2BGR))
        depth_imgs.append(depth)
    return color_imgs, depth_imgs


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default="./sample/dennis_normalized.ply")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_cameras", default=8)
    args = parser.parse_args()

    raw_scan = trimesh.load(args.mesh)
    # normalize to 0~1 for the Hash encoding
    raw_scan.vertices += 0.5
    meshes = [raw_scan]

    os.makedirs("outputs/image", exist_ok=True)
    os.makedirs("outputs/mask", exist_ok=True)
    cameras = get_cameras(args.num_cameras)
    render_imgs, render_depths = render_image(cameras, meshes)
    for i, (render_img, render_depth) in enumerate(zip(render_imgs, render_depths)):
        render_mask = np.repeat(render_depth[:, :, None] > 0, 3, axis=2).astype(np.uint8) * 255
        img = np.concatenate([render_img, render_mask], axis=1)
        cv2.imwrite(f"outputs/image/image_{i:02d}.png", render_img)
        cv2.imwrite(f"outputs/mask/mask_{i:02d}.png", render_mask)
        if args.visualize:
            cv2.imshow(f"View", render_img)
            cv2.waitKey()

    d = {}
    for i, (camera, extrinsic) in enumerate(cameras): 
        K = np.eye(4)
        K[0, 0] = camera.fx
        K[1, 1] = camera.fy
        K[0, 2] = camera.cx
        K[1, 2] = camera.cy
        P = K @ extrinsic
        d[f"cam_{i}"] = P
    np.savez("outputs/cameras.npz", **d)
