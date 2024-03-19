# Copyright (C) 2023  ETH Zurich, Manuel Kaufmann, Velko Vechev, Dario Mylonopoulos
import os
import re
import argparse
from pathlib import Path
from dataclasses import dataclass

from typing import Dict, Any, Tuple, List

import numpy as np
import cv2 as cv
import torch
from tqdm import tqdm
import json5 as json
import trimesh

from aitviewer.configuration import CONFIG as C
from aitviewer.models.smpl import SMPLLayer
from aitviewer.renderables.billboard import Billboard
from aitviewer.renderables.smpl import SMPLSequence
from aitviewer.scene.camera import OpenCVCamera
from aitviewer.viewer import Viewer
from aitviewer.headless import HeadlessRenderer
from aitviewer.renderables.plane import ChessboardPlane

HOME_DIR = os.path.expanduser("~")
SMPL_MODEL_DIR = os.path.join(HOME_DIR, "smpl-model-data")
# C.smplx_models = SMPL_MODEL_DIR
C.update_conf({"smplx_models": '/media/ubuntu/hdd/Motion_infilling_smoothing/SmoothNet/data/'})
C.auto_set_floor = False
C.z_up = True

COLORS = [
    (0.2, 0.4, 0.6, 1.0),
    (0.8, 0.1, 0.3, 1.0),
    (0.5, 0.7, 0.2, 1.0),
    (0.9, 0.5, 0.1, 1.0),
    (0.3, 0.6, 0.8, 1.0),
    (0.7, 0.2, 0.4, 1.0),
    (0.1, 0.9, 0.6, 1.0),
    (0.4, 0.8, 0.3, 1.0),
    (0.6, 0.3, 0.7, 1.0),
    (0.2, 0.5, 0.9, 1.0),
    (0.8, 0.4, 0.1, 1.0),
    (0.9, 0.7, 0.2, 1.0),
    (0.5, 0.2, 0.6, 1.0),
    (0.3, 0.8, 0.4, 1.0),
    (0.7, 0.1, 0.9, 1.0),
    (0.6, 0.5, 0.3, 1.0),
    (0.4, 0.6, 0.8, 1.0),
    (0.1, 0.7, 0.2, 1.0),
    (0.9, 0.3, 0.5, 1.0),
    (0.2, 0.8, 0.7, 1.0),
    (0.7, 0.9, 0.4, 1.0),
    (0.5, 0.4, 0.1, 1.0),
    (0.3, 0.6, 0.9, 1.0),
    (0.8, 0.2, 0.5, 1.0),
    (0.1, 0.5, 0.8, 1.0),
    (0.4, 0.9, 0.2, 1.0),
    (0.6, 0.1, 0.7, 1.0),
    (0.9, 0.8, 0.3, 1.0),
    (0.2, 0.3, 0.5, 1.0),
    (0.7, 0.5, 0.6, 1.0),
]

# DEBUG_FRAME_LIMIT = 50
# DEBUG_PERSON_LIMIT = 5
DEBUG_FRAME_LIMIT = -1
DEBUG_PERSON_LIMIT = -1


def build_parser() -> argparse.ArgumentParser:
    """
    Build the argument parser.

    Returns:
        Argument parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--world-res", type=Path, help="Path to the world results NPZ file."
    )
    parser.add_argument("--frames", type=Path, help="Path to the frames directory.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Interactive mode. If false, save the visualization to a file.",
    )
    parser.add_argument(
        "--reproj",
        action="store_true",
        help="Reproject SMPL. If false, show SMPL in world coordinates.",
    )
    parser.add_argument(
        "--static-view",
        type=Path,
        help="Path to static view JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Path to the output video file. Only used if interactive is false.",
    )
    parser.add_argument(
        "--outputSMPL",
        type=Path,
        help="Path to the output SMPL file. Only used if interactive is false.",
    )
    return parser


@dataclass
class Args:
    world_res: Path
    frames: Path
    interactive: bool
    reproj: bool
    static_view: Path
    output: Path
    outputSMPL: Path


def parse_args() -> Args:
    parser = build_parser()
    args = parser.parse_args()
    return Args(
        world_res=args.world_res,
        frames=args.frames,
        interactive=args.interactive,
        reproj=args.reproj,
        static_view=args.static_view,
        output=args.output,
        outputSMPL=args.outputSMPL,
    )


def get_image_dims(images_path: Path) -> Tuple[int, int]:
    """
    Returns the dimensions of an image.

    Args:
        path: Path to folder containing images.

    Returns:
        Tuple containing the width and height of the image.
    """
    img = cv.imread(str(images_path / os.listdir(images_path)[0]))
    return img.shape[:2][::-1]
    # return 1920, 1080


def build_billboard(
    camera: OpenCVCamera,
    frames_path: Path,
    n_frames: int,
) -> Billboard:
    """
    Create a billboard from a set of images.

    Args:
        camera: Camera to use for the billboard.
        frames_path: Path to folder containing images.
        n_frames: Number of frames to use for the billboard.

    Returns:
        Billboard object.
    """
    img_width, img_height = get_image_dims(frames_path)

    # Sort images by frame number in the filename.
    regex = re.compile(r"(\d*)$")

    def sort_key(x):
        name = os.path.splitext(x)[0]
        return int(regex.search(name).group(0))

    sorted_frame_names = sorted(os.listdir(frames_path), key=sort_key)
    images_paths = [str(frames_path / name) for name in sorted_frame_names[:n_frames]]

    billboard = Billboard.from_camera_and_distance(
        camera,
        100.0,
        cols=img_width,
        rows=img_height,
        textures=images_paths[:DEBUG_FRAME_LIMIT]
        if DEBUG_FRAME_LIMIT > 0
        else images_paths,
    )

    return billboard


def rotx(theta: float) -> np.ndarray:
    """
    Create a rotation matrix around the x-axis.

    Args:
        theta: Rotation angle in radians.

    Returns:
        Rotation matrix.
    """
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ]
    )


def roty(theta: float) -> np.ndarray:
    """
    Create a rotation matrix around the y-axis.

    Args:
        theta: Rotation angle in radians.

    Returns:
        Rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ]
    )


def rotz(theta: float) -> np.ndarray:
    """
    Create a rotation matrix around the z-axis.

    Args:
        theta: Rotation angle in radians.

    Returns:
        Rotation matrix.
    """
    return np.array(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def deg2rad(deg: float) -> float:
    """
    Convert degrees to radians.

    Args:
        deg: Angle in degrees.

    Returns:
        Angle in radians.
    """
    return deg * np.pi / 178.0


def load_static_cam(
    static_view: Path,
    img_width: int,
    img_height: int,
) -> OpenCVCamera:
    """
    Load static view from JSON.

    Args:
        static_view: Path to static view JSON.
        img_width: Width of the images.
        img_height: Height of the images.
    """
    with open(static_view, "r", encoding="utf-8") as f:
        data = json.load(f)

    R_c = (
        roty(deg2rad(data["roty"]))
        @ rotx(deg2rad(data["rotx"]))
        @ rotz(deg2rad(data["rotz"]))
    )
    cop = np.array(data["cop"])

    c2w = np.eye(4)
    c2w[:3, :3] = R_c
    c2w[:3, 3] = cop

    w2c = np.linalg.inv(c2w)

    K = np.eye(3)
    K[0, 0] = data["focal_length"]
    K[1, 1] = data["focal_length"]
    K[0, 2] = img_width / 2.0
    K[1, 2] = img_height / 2.0

    static_camera = OpenCVCamera(
        K=K,
        Rt=w2c[:3, :],
        cols=img_width,
        rows=img_height,
        name="Static View Camera",
    )

    return static_camera


def load_cam_data(data: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera extrinsics and intrinsics from the output of the SLAHMR.

    Args:
        data: Dictionary containing the SLAHMR data.

    Returns:
        Camera extrinsics (n_frames, 4, 4) and intrinsics (n_frames, 3, 3).
    """
    # Transform y_up coordinates to z_up coordinates.
    # z_up_from_y_up = np.array(
    #     [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], np.float32
    # ).T

    R = data["cam_R"][0]  # (n_frames, 3, 3)
    t = data["cam_t"][0]  # (n_frames, 3)

    R_c = R.transpose((0, 2, 1))  # (n_frames, 3, 3)
    COP = -R_c @ t[..., np.newaxis]  # (n_frames, 3, 1)

    R_c = rotx(np.pi) @ R_c  # (n_frames, 3, 3)
    COP = rotx(np.pi)[np.newaxis, ...] @ COP  # (n_frames, 3, 1)

    R = R_c.transpose((0, 2, 1))  # (n_frames, 3, 3)
    t = -R @ COP  # (n_frames, 3)
    t = t[:, :, 0]  # (n_frames, 3)

    # R = rotx(np.pi) @ R  # (n_frames, 3, 3)
    # t = rotx(np.pi)[np.newaxis, ...] @ t[..., np.newaxis]  # (n_frames, 3, 1)
    # t = t[:, :, 0]  # (n_frames, 3)

    Rt = np.zeros((R.shape[0], 4, 4), dtype=np.float32)
    Rt[:, :3, :3] = R
    Rt[:, :3, 3] = t

    K = np.zeros((R.shape[0], 3, 3), dtype=np.float32)
    K[:, 0, 0] = data["intrins"][0]
    K[:, 1, 1] = data["intrins"][1]
    K[:, 0, 2] = data["intrins"][2]
    K[:, 1, 2] = data["intrins"][3]
    K[:, 2, 2] = 1.0

    return Rt, K


def load_smpl_seqs(data: Dict[str, Any], n_frames: int) -> List[SMPLSequence]:
    """
    Load SMPL sequences from the output of the SLAHMR.

    Args:
        data: Dictionary containing the SLAHMR data.
        n_frames: Number of frames to use for the SMPL sequences.

    Returns:
        List of SMPL sequences.
    """
    smpl_sequences = []
    n_people = data["trans"].shape[0]

    if DEBUG_PERSON_LIMIT > 0:
        n_people = min(n_people, DEBUG_PERSON_LIMIT)

    for p in tqdm(range(n_people), desc="Loading SMPL sequences"):
        if "track_mask" not in data:
            valid_idxs = np.arange(n_frames)
        else:
            valid_idxs = data["track_mask"][p].nonzero()[0]
        # padding valid idx to n_frames
        valid_idxs = np.pad(valid_idxs, (0, n_frames - len(valid_idxs)), "edge")
        # import pdb;pdb.set_trace()
        print(valid_idxs)

        pose = data["pose_body"][p][valid_idxs]
        # Need to pad to 23 joints for AITViewer. SLAHMR only has 21. Possible
        # reason for missing joints could be:
        # https://files.is.tue.mpg.de/black/talks/SMPL-made-simple-FAQs.pdf
        # - see slide 7.
        pose = np.pad(pose, ((0, 0), (0, 6)))

        betas = data["betas"][p, :10]
        betas = np.repeat(betas[np.newaxis, :], n_frames, axis=0)
        betas = betas[valid_idxs]

        trans = data["trans"][p][valid_idxs]
        # trans = (
        #     rotx(np.pi)[np.newaxis, ...] @ trans[..., np.newaxis]
        # )  # (n_frames, 3, 1)
        # trans = trans[:, :, 0]  # (n_frames, 3)

        ori = data["root_orient"][p][valid_idxs]  # (n_frames, 3)
        # ori = rotx(np.pi) @ aa2rot(ori)  # (n_frames, 3, 3)
        # ori = rot2aa(ori)  # (n_frames, 3)

        # enabled_frames = np.zeros(n_frames, dtype=bool)
        # enabled_frames = np.zeros(n_frames, dtype=bool)
        # enabled_frames[valid_idxs] = True
        enabled_frames = np.ones(n_frames, dtype=bool)

        if DEBUG_FRAME_LIMIT > 0:
            # valid_idxs = valid_idxs[valid_idxs < DEBUG_FRAME_LIMIT]
            pose = pose[:DEBUG_FRAME_LIMIT]
            betas = betas[:DEBUG_FRAME_LIMIT]
            trans = trans[:DEBUG_FRAME_LIMIT]
            ori = ori[:DEBUG_FRAME_LIMIT]
            enabled_frames = enabled_frames[:DEBUG_FRAME_LIMIT]

        # enabled_frames[valid_idxs] = True

        def post_fk_func(
            self: SMPLSequence,
            vertices: torch.Tensor,
            joints: torch.Tensor,
            current_frame_only: bool,
        ):
            """
            Define a postprocess function for the SMPL sequence,
            we use this to apply the translation from the data to the root node.
            """
            # Select the translation of the current frame if
            # current_frame_only is True, otherwise select all frames.
            # t = trans[[self.current_frame_id]] if current_frame_only else trans[:]
            # t = torch.from_numpy(t).to(dtype=joints.dtype, device=joints.device)

            # # Subtract the position of the root joint from all vertices and
            # # joint positions and add the root translation.
            # cur_root_trans = joints[:, [0], :]
            # vertices = vertices - cur_root_trans + t[:, None, :]
            # joints = joints - cur_root_trans + t[:, None, :]

            rot = (
                torch.from_numpy(rotx(np.pi)[None, None, ...])
                .float()
                .to(vertices.device)
            )

            vertices = rot @ vertices[..., None]  # (n_frames, 6890, 3, 1)
            vertices = vertices[:, :, :, 0]  # (n_frames, 6890, 3)

            joints = rot @ joints[..., None]  # (n_frames, 45, 3, 1)
            joints = joints[:, :, :, 0]  # (n_frames, 45, 3)

            return vertices, joints

        smpl_layer = SMPLLayer(model_type="smpl", gender="neutral", device=C.device)
        smpl_sequence = SMPLSequence(
            poses_body=pose,
            poses_root=ori,
            betas=betas,
            trans=trans,
            is_rigged=False,
            smpl_layer=smpl_layer,
            color=COLORS[p % len(COLORS)],
            # z_up=False,
            post_fk_func=post_fk_func,
            name=f"Player {p}",
            enabled_frames=enabled_frames,
        )

        smpl_sequences.append(smpl_sequence)

    return smpl_sequences


def show_reproj_vis(
    smpl_sequences: List[SMPLSequence],
    camera: OpenCVCamera,
    billboard: Billboard,
    img_width: int,
    img_height: int,
) -> None:
    """
    Visualize the output reprojection interactively.

    Args:
        smpl_sequences: List of SMPL sequences.
        camera: Camera to use for the visualization.
        billboard: Billboard to use for the visualization.
        img_width: Width of the images.
        img_height: Height of the images.
    """
    viewer = Viewer(size=(img_width, img_height))

    camera.viewer = viewer

    viewer.scene.add(*smpl_sequences, billboard, camera)

    viewer.scene.floor.enabled = False
    viewer.scene.fps = 50.0
    viewer.playback_fps = 50.0
    # viewer.shadows_enabled = False

    viewer.set_temp_camera(camera)

    viewer.run()


def show_world_vis(
    smpl_sequences: List[SMPLSequence],
    camera: OpenCVCamera,
    floor_pose: np.ndarray,
    img_width: int,
    img_height: int,
) -> None:
    """
    Visualize the output in world coordinates interactively.

    Args:
        smpl_sequences: List of SMPL sequences.
        camera: Camera to use for the visualization.
        img_width: Width of the images.
        img_height: Height of the images.
    """
    viewer = Viewer(size=(img_width, img_height))

    camera.viewer = viewer

    new_floor = ChessboardPlane(
        100.0,
        200,
        (0.9, 0.9, 0.9, 1.0),
        (0.82, 0.82, 0.82, 1.0),
        "xy" if C.z_up else "xz",
        name="SLAHMR Floor",
    )
    new_floor.material.diffuse = 0.1

    floor_rotations = floor_pose[np.newaxis, :3, :3]  # (1, 3, 3)
    floor_positions = floor_pose[np.newaxis, :3, 3, np.newaxis]  # (1, 3, 1)

    rot = rotx(np.pi)[None, ...]  # (1, 3, 3)

    floor_rotations = rot @ floor_rotations  # (1, 3, 3)
    floor_positions = rot @ floor_positions  # (1, 3, 1)
    floor_positions = floor_positions[:, :, 0]  # (1, 3)

    new_floor._rotations = floor_rotations
    new_floor._positions = floor_positions

    viewer.scene.add(*smpl_sequences, camera, new_floor)

    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.scene.fps = 50.0
    viewer.playback_fps = 50.0
    # viewer.shadows_enabled = False

    viewer.set_temp_camera(camera)

    viewer.run()


def show_world_vis_static(
    smpl_sequences: List[SMPLSequence],
    camera: OpenCVCamera,
    static_camera: OpenCVCamera,
    floor_pose: np.ndarray,
    img_width: int,
    img_height: int,
) -> None:
    """
    Visualize the output in world coordinates interactively from static view.

    Args:
        smpl_sequences: List of SMPL sequences.
        camera: Camera in output.
        static_camera: Static camera to use for the visualization.
        img_width: Width of the images.
        img_height: Height of the images.
    """
    viewer = Viewer(size=(img_width, img_height))

    camera.viewer = viewer
    static_camera.viewer = viewer

    viewer.scene.add(*smpl_sequences, camera, static_camera)

    # new_floor = ChessboardPlane(
    #     100.0, 200, (0.9, 0.9, 0.9, 1.0), (0.82, 0.82, 0.82, 1.0), name="New Floor"
    # )
    new_floor = ChessboardPlane(
        100.0,
        200,
        (0.9, 0.9, 0.9, 1.0),
        (0.82, 0.82, 0.82, 1.0),
        "xy" if C.z_up else "xz",
        name="New Floor",
    )
    new_floor.material.diffuse = 0.1
    # new_floor = ChessboardPlane(100.0, 200, name="New Floor")

    floor_rotations = floor_pose[np.newaxis, :3, :3]  # (1, 3, 3)
    floor_positions = floor_pose[np.newaxis, :3, 3, np.newaxis]  # (1, 3, 1)

    rot = rotx(np.pi)[None, ...]  # (1, 3, 3)

    floor_rotations = rot @ floor_rotations  # (1, 3, 3)
    floor_positions = rot @ floor_positions  # (1, 3, 1)
    floor_positions = floor_positions[:, :, 0]  # (1, 3)

    new_floor._rotations = floor_rotations
    new_floor._positions = floor_positions

    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.scene.fps = 50.0
    viewer.playback_fps = 50.0
    # viewer.shadows_enabled = False

    # viewer.scene.lights[0].shadow_enabled = True
    # viewer.scene.lights[0].shadow_map_size = 63
    # viewer.scene.lights[0].shadow_map_near = 0.01
    # viewer.scene.lights[0].shadow_map_far = 50

    # # fix light
    # viewer.scene.lights[0].azimuth = 90
    # viewer.scene.lights[1].azimuth = 90

    # # fix shadow map
    # viewer.scene.lights[0].elevation = 0
    # viewer.scene.lights[0].position = (0, 0, -30)

    # viewer.set_temp_camera(static_camera)
    viewer.set_temp_camera(camera)

    # viewer.scene.floor._rotations = floor_rotations
    # viewer.scene.floor._positions = floor_positions

    viewer.scene.add(new_floor)

    viewer.run()


def render_reproj_vis(
    smpl_sequences: List[SMPLSequence],
    camera: OpenCVCamera,
    billboard: Billboard,
    img_width: int,
    img_height: int,
    output: Path,
) -> None:
    """
    Render video of the output reprojection.

    Args:
        smpl_sequences: List of SMPL sequences.
        camera: Camera to use for the visualization.
        billboard: Billboard to use for the visualization.
        img_width: Width of the images.
        img_height: Height of the images.
        output: Path to the output video file.
    """
    viewer = HeadlessRenderer(size=(img_width, img_height))

    camera.viewer = viewer

    viewer.scene.add(*smpl_sequences, billboard, camera)

    viewer.scene.floor.enabled = False
    viewer.scene.origin.enabled = False
    viewer.scene.fps = 50.0
    viewer.playback_fps = 50.0
    viewer.shadows_enabled = True

    viewer.set_temp_camera(camera)

    viewer.save_video(
        video_dir=str(output),
        output_fps=50.0,
        ensure_no_overwrite=False,
        quality="high",
    )


def render_world_vis(
    smpl_sequences: List[SMPLSequence],
    camera: OpenCVCamera,
    img_width: int,
    img_height: int,
    output: Path,
) -> None:
    """
    Render video of the output in world coordinates.

    Args:
        smpl_sequences: List of SMPL sequences.
        camera: Camera to use for the visualization.
        img_width: Width of the images.
        img_height: Height of the images.
        output: Path to the output video file.
    """
    viewer = HeadlessRenderer(size=(img_width, img_height))

    camera.viewer = viewer

    viewer.scene.add(*smpl_sequences, camera)

    viewer.scene.floor.enabled = True
    viewer.scene.origin.enabled = False
    viewer.scene.fps = 50.0
    viewer.playback_fps = 50.0
    viewer.shadows_enabled = True

    viewer.set_temp_camera(camera)

    viewer.save_video(
        video_dir=str(output),
        output_fps=50.0,
        ensure_no_overwrite=False,
        quality="high",
    )


def render_world_vis_static(
    smpl_sequences: List[SMPLSequence],
    camera: OpenCVCamera,
    static_camera: OpenCVCamera,
    img_width: int,
    img_height: int,
    output: Path,
) -> None:
    """
    Render video of the output in world coordinates from static view.

    Args:
        smpl_sequences: List of SMPL sequences.
        camera: Camera to use for the visualization.
        img_width: Width of the images.
        img_height: Height of the images.
        output: Path to the output video file.
    """
    viewer = HeadlessRenderer(size=(img_width, img_height))

    camera.viewer = viewer
    static_camera.viewer = viewer

    viewer.scene.add(*smpl_sequences, camera)

    viewer.scene.floor.enabled = True
    viewer.scene.origin.enabled = False
    viewer.scene.fps = 50.0
    viewer.playback_fps = 50.0
    viewer.shadows_enabled = True

    viewer.set_temp_camera(static_camera)

    viewer.save_video(
        video_dir=str(output),
        output_fps=50.0,
        ensure_no_overwrite=False,
        quality="high",
    )


def pose_from_normal_vector(N: np.ndarray, d: float) -> np.ndarray:
    """
    Create a pose matrix from a normal vector and a distance.

    Args:
        N: Normal vector.
        d: Distance.

    Returns:
        Pose matrix, 4x4.
    """
    # Normalize the normal vector
    N = N / np.linalg.norm(N)

    # Compute the rotation matrix to align the z-axis with the normal vector
    z_axis = np.array([0, -1, 0])
    v = np.cross(z_axis, N)
    s = np.linalg.norm(v)
    c = np.dot(z_axis, N)
    skew_matrix = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = (
        np.eye(3) + skew_matrix + np.dot(skew_matrix, skew_matrix) * (1 - c) / (s**2)
    )

    # Create the 4x4 pose matrix
    pose_matrix = np.eye(4)
    pose_matrix[:3, :3] = rotation_matrix
    pose_matrix[:3, 3] = N * d  # Adjusted line

    return pose_matrix

def word2cam(data_world, proj_matrix):
    """
    Project 3d world coordinates to 3d camera coordinates
    data_world: (B, N, 3)
    proj_matrix: (B, 3, 4)
    return
    data_cam: (B, N, 3)
    """
    ext_arr = np.ones((data_world.shape[0], data_world.shape[1], 1))
    data_homo = np.concatenate((data_world, ext_arr), axis=2)
    # data_cam = data_homo @ proj_matrix.T
    data_cam = np.einsum("bnc,bdc->bnd", data_homo, proj_matrix)
    return data_cam


def main() -> None:
    """Main function."""
    args = parse_args()
    if args.output is not None:
        os.makedirs(args.output.parent, exist_ok=True)
    
    if args.outputSMPL is not None:
        os.makedirs(args.outputSMPL, exist_ok=True)
    # Load camera and SMPL data from the output of the GLAMR dynamic camera
    # demo from https://github.com/NVlabs/GLAMR.
    data = dict(np.load(args.world_res))

    n_frames = data["trans"].shape[1]

    Rt, K = load_cam_data(data)

    # Create a sequence of cameras from camera extrinsics and intrinsics.
    img_width, img_height = get_image_dims(args.frames)
    # print(img_width, img_height)
    # img_width, img_height = 470, 640
    camera = OpenCVCamera(
        K=K, Rt=Rt[:, :3, :], cols=img_width, rows=img_height, name="Scene Camera"
    )

    smpl_sequences = load_smpl_seqs(data, n_frames)

    os.makedirs(f"{args.frames}/../test_mesh/0", exist_ok=True)
    os.makedirs(f"{args.frames}/../test_mesh/1", exist_ok=True)
    joints_cam_list = []
    joints_2d_list = []
    verts_list = []
    for person_i, smpl_seq in enumerate(smpl_sequences):
        # print(smpl_seq.vertices.shape)
        # print(smpl_seq.joints.shape)
        # print(smpl_seq.faces.shape)
        # print(Rt.shape)
        # print(K.shape)
        verts_cam = word2cam(smpl_seq.vertices, Rt[:smpl_seq.vertices.shape[0], :3, :])
        joints_cam = word2cam(smpl_seq.joints, Rt[:smpl_seq.joints.shape[0], :3, :])
        joints_2d = np.einsum("bij,bnj->bni",K[:smpl_seq.joints.shape[0], ...], joints_cam)
        joints_2d = joints_2d / joints_2d[..., 2:]
        joints_2d = joints_2d[..., :2]
        # print(joints_2d[0])
        joints_cam_list.append(joints_cam)
        joints_2d_list.append(joints_2d)
        verts_list.append(verts_cam)
        # import pdb;pdb.set_trace()
        for frame_i in range(smpl_seq.vertices.shape[0]):
            mesh_pred = trimesh.Trimesh(vertices = verts_cam[frame_i], faces = smpl_seq.faces)
            mesh_pred.export(f"{args.frames}/../test_mesh/{person_i}/{frame_i:04d}_deformed.ply")

            
        # print(joints_cam.shape)
    print(np.array(joints_2d_list).shape)
    print(np.array(joints_cam_list).shape)
    print(np.array(verts_list).shape)
    # np.savez(f'{args.frames}/../all', pj2d_org=np.array(joints_2d_list), joints=np.array(joints_cam_list), verts=np.array(verts_list), cam_trans=np.zeros((2, n_frames, 3)))
    # add face to evaluate the mesh


    if "floor_plane" not in data:
        floor_offset = 0
        floor_normal = np.array([0, 1, 0])
    else:
        floor_plane = data["floor_plane"][0]
        floor_offset = np.linalg.norm(floor_plane)
        floor_normal = floor_plane / floor_offset
    # floor_offset = 0
    # floor_normal = np.array([0, 1, 0])

    floor_pose = pose_from_normal_vector(floor_normal, floor_offset)

    if args.reproj:
        billboard = build_billboard(
            camera=camera,
            frames_path=args.frames,
            n_frames=n_frames,
        )
        if args.interactive:
            show_reproj_vis(
                smpl_sequences=smpl_sequences,
                camera=camera,
                billboard=billboard,
                img_width=img_width,
                img_height=img_height,
            )
        else:
            render_reproj_vis(
                smpl_sequences=smpl_sequences,
                camera=camera,
                billboard=billboard,
                img_width=img_width,
                img_height=img_height,
                output=args.output,
            )
    else:
        if args.interactive:
            if args.static_view is not None:
                # w2cs, _, _, _ = load_vizrt_calibration(source=args.calibration)
                # w2cs = np.array([w2cs[frame] for frame in sorted(w2cs.keys())])
                # if DEBUG_FRAME_LIMIT > 0:
                #     w2cs = w2cs[:DEBUG_FRAME_LIMIT]
                # c2ws = np.linalg.inv(w2cs)
                # avg_c2w = np.mean(c2ws, axis=0)
                # full_avg_c2ws = np.repeat(avg_c2w[None, :, :], w2cs.shape[0], axis=0)
                # full_avg_w2cs = np.linalg.inv(full_avg_c2ws)
                # static_camera = OpenCVCamera(
                #     K=K,
                #     Rt=full_avg_w2cs[:, :3, :],
                #     cols=img_width,
                #     rows=img_height,
                #     name="Static View Camera",
                # )
                static_camera = load_static_cam(
                    static_view=args.static_view,
                    img_width=img_width,
                    img_height=img_height,
                )

                import ipdb

                ipdb.set_trace()
                show_world_vis_static(
                    smpl_sequences=smpl_sequences,
                    camera=camera,
                    static_camera=static_camera,
                    floor_pose=floor_pose,
                    img_width=img_width,
                    img_height=img_height,
                )
                return
            show_world_vis(
                smpl_sequences=smpl_sequences,
                camera=camera,
                floor_pose=floor_pose,
                img_width=img_width,
                img_height=img_height,
            )
        else:
            if args.static_view is not None:
                static_camera = load_static_cam(
                    static_view=args.static_view,
                    img_width=img_width,
                    img_height=img_height,
                )

                render_world_vis_static(
                    smpl_sequences=smpl_sequences,
                    camera=camera,
                    static_camera=static_camera,
                    img_width=img_width,
                    img_height=img_height,
                    output=args.output,
                )
                return
            render_world_vis(
                smpl_sequences=smpl_sequences,
                camera=camera,
                img_width=img_width,
                img_height=img_height,
                output=args.output,
            )



if __name__ == "__main__":
    main()
