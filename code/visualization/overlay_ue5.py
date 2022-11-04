import numpy as np
import cv2
import trimesh
from scipy.spatial.transform import Rotation as R

focal_length = -1920 / 10.
cam_intrinsics = np.array([[focal_length, 0., 960.],[0.,focal_length, 540.],[0.,0.,1.]])
K = np.eye(4)
K[:3, :3] = cam_intrinsics

extrinsic = np.eye(4)
cam_center = np.array([-608.898933, 79.143338, 114.120324]) / 100. # np.array([-526.903522, 61.717835, 115.28066]) / 100.
rotation_angle = np.array([0.800000, -9.582521, 0.000000]) * np.pi / 180.
rotation_angle = R.from_rotvec(rotation_angle)

cam_R = rotation_angle.as_matrix()
cam_T = -cam_R @ cam_center
extrinsic[:3, :3] = cam_R
extrinsic[:3, 3] = cam_T
# extrinsic[1:3] *= -1.
P = K @ extrinsic

mesh = trimesh.load('/home/chen/disk2/RGB_PINA_MoCap/00027_Phonecall/meshes_vis/mesh-f00040.obj')
img = cv2.imread('/home/chen/disk2/Youtube_Videos/00027_Phonecall/frames/000037.png')
for j in range(0, mesh.vertices.shape[0]):
    padded_v = np.pad(mesh.vertices[j]*1000., (0,1), 'constant', constant_values=(0,1))
    temp = P @ padded_v.T # np.load('/home/chen/snarf_idr_cg_1/data/buff_new/cameras.npz')['cam_0'] @ padded_v.T
    pix = (temp/temp[2])[:2]
    # import ipdb; ipdb.set_trace()
    output_img = cv2.circle(img, tuple(pix.astype(np.int32)), 3, (0,255,255), -1)

cv2.imwrite('/home/chen/Desktop/test_projcam_UE5.png', output_img)