import pyrender
import numpy as np
import cv2
import trimesh
import os

WIDTH = 480
HEIGHT = 640
FOCAL = 1000
smpl_faces = np.load('/home/chen/IPNet/faces.npy')
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
    
    
def get_camera_rig(num_cameras, axis=(0, 1, 0), C=(0, 0, 1), at=(0, 0, 0)):
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
    for C in [(0, 1.0, 3)]: # , (0, 1.5, 3), (0, 2.0, 3)
        for extrinsic in get_camera_rig(num_cameras, C=C):
            camera = pyrender.IntrinsicsCamera(fx=FOCAL,
                                               fy=FOCAL,
                                               cx=WIDTH / 2,
                                               cy=HEIGHT / 2)
            cameras.append((camera, extrinsic))
    return cameras


def render_image(cameras, meshes):
    scene = pyrender.Scene(ambient_light=[0.7, 0.7, 0.7])
    lights = [pyrender.DirectionalLight() for _ in range(2)]
    lposes = [trimesh.transformations.translation_matrix(b) for b in scene.bounds]

    for light, lpose in zip(lights, lposes):
        scene.add(light, pose=lpose)
    for mesh in meshes:
        mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
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
    import glob
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mesh", type=str, default="./sample/dennis_normalized.ply")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--num_cameras", default=8)
    args = parser.parse_args()
    
    # raw_scan = trimesh.load(args.mesh)
    files = sorted(glob.glob('/home/chen/disk2/cape_release/sequences/00096/shortlong_hips/*.npz'))
    scans = sorted(glob.glob('/home/chen/disk2/cape_release/raw_scans_textured/00096/shortlong_hips/*.ply'))
    pose = []
    trans = []
    skip_step = 1
    normalize_shift = 0.
    # end_idx = 100 # exclude the later frames due to the dirty data
    # import ipdb 
    # ipdb.set_trace()
    files = files[::skip_step]# [:end_idx]
    scans = scans[::skip_step]# [:end_idx]
    args.num_cameras = len(files)
    for idx, file_path in enumerate(files):
        # file_path = '/home/chen/disk2/cape_release/sequences/03375/blazerlong_babysit_trial1/blazerlong_babysit_trial1.000001.npz'
        file = np.load(file_path)
        frame = int(file_path.split('.')[-2])

        raw_scan = trimesh.load(scans[idx]) # trimesh.Trimesh(file['v_posed']-file['transl'], smpl_faces)
        raw_scan.apply_scale(1/1000)
        raw_scan = trimesh.graph.split(raw_scan, only_watertight=False)[0]
        
        # normalize meshes within the unit sphere
        if idx == 0:
            max = raw_scan.vertices.max(axis=0)
            min = raw_scan.vertices.min(axis=0)
            normalize_shift = -(max + min) / 2
            raw_scan.vertices = (raw_scan.vertices + normalize_shift) # / (max - min).max() * 2
        else:
            raw_scan.vertices = (raw_scan.vertices + normalize_shift)
        trans.append(normalize_shift+file['transl'])
        meshes = [raw_scan]
        
        pose.append(file['pose'])
        cameras = get_cameras(args.num_cameras)

        render_imgs, render_depths = render_image(cameras[idx:idx+1], meshes)
        for i, (render_img, render_depth) in enumerate(zip(render_imgs, render_depths)):
            render_mask = np.repeat(render_depth[:, :, None] > 0, 3, axis=2).astype(np.uint8) * 255
            img = np.concatenate([render_img, render_mask], axis=1)
            os.makedirs(f"outputs/image", exist_ok=True)
            os.makedirs(f"outputs/mask", exist_ok=True)
            cv2.imwrite(f"outputs/image/image_{frame:04d}.png", render_img)
            cv2.imwrite(f"outputs/mask/mask_{frame:04d}.png", render_mask)
            
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
    np.save("outputs/poses.npy", np.array(pose))
    np.save("outputs/normalize_trans.npy", np.array(trans))