import cv2
import trimesh
import numpy as np
import glob
seq = 'Pablo_outdoor'
DIR = f'/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/{seq}'
with open(f'{DIR}/calib.txt') as f:
    lines = f.readlines()

cam_params = lines[2].split()
cam_intrinsics = np.array([[float(cam_params[1]), 0., float(cam_params[3]), 0.], 
                           [0., float(cam_params[6]), float(cam_params[7]), 0.], 
                           [0., 0., 1., 0.],
                           [0., 0., 0., 1.]])
cam_extrinsics = np.array(lines[3].split()[1:]).astype(np.float32).reshape(4,4)
P = cam_intrinsics @ cam_extrinsics

gt_mesh_paths = sorted(glob.glob(f'{DIR}/ground_truth/*.off'))
image_paths= sorted(glob.glob(f'{DIR}/frames/*.png'))

for idx, gt_mesh_path in enumerate(gt_mesh_paths):
    gt_mesh = trimesh.load(gt_mesh_path, process=False)
    image = cv2.imread(image_paths[idx])
    for j in range(0, gt_mesh.vertices.shape[0]):
        padded_v = np.pad(gt_mesh.vertices[j], (0,1), 'constant', constant_values=(0,1))
        temp = P @ padded_v.T
        pix = (temp/temp[2])[:2]
        output_img = cv2.circle(image, tuple(pix.astype(np.int32)), 3, (0,255,255), -1)
    
    cv2.imwrite(f'/home/chen/disk2/MPI_INF_Dataset/MonoPerfCapDataset/Pablo_outdoor/overlay/{idx:04d}.png', output_img)
