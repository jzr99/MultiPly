import argparse
import cv2
import json
import numpy as np
import os
import pickle

from tqdm import tqdm
import trimesh
import glob

def main(args):
    root = args.capture_root
    mesh_root = os.path.join(root, 'meshes')
    mesh_output_dir = os.path.join(root, 'meshes_vis')
    # v_normal_output_dir = os.path.join(root, 'v_normal')
    if not os.path.exists(mesh_output_dir):
        os.makedirs(mesh_output_dir)
    mesh_paths = sorted(glob.glob(f"{mesh_root}/*.npz"))
    for mesh_path in mesh_paths:
        mesh_dict = np.load(mesh_path)

        mesh = trimesh.Trimesh(vertices=mesh_dict['vertices'], faces=mesh_dict['faces'], process=False)
        mesh_output_path = os.path.join(mesh_output_dir, os.path.basename(mesh_path).replace('.npz', '.ply'))
        _ = mesh.export(mesh_output_path)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--capture_root', required=True)
    main(parser.parse_args())