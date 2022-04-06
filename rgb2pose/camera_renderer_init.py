import json
from smpl_rendering.camera import PerspectiveCamera
from smpl_rendering.smpl_renderer import SMPLRenderer
import numpy as np
import torch
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def cam_renderer_init(batch_size=1, cam_intrinsic=None, img_size=[1080, 1920], device='cuda:0', gender='male', T=None):

    # with open('/home/chen/kinect_capture/camera.json') as f:
    #     data = json.load(f)
    # color_camera = data["CalibrationInformation"]["Cameras"][1]

    # intrinsic_params = color_camera["Intrinsics"]["ModelParameters"]

    # cx = intrinsic_params[0]
    # cy = intrinsic_params[1]
    # fx = intrinsic_params[2]
    # fy = intrinsic_params[3]
    # k1 = intrinsic_params[4]
    # k2 = intrinsic_params[5]
    # k3 = intrinsic_params[6]
    # k4 = intrinsic_params[7]
    # k5 = intrinsic_params[8]
    # k6 = intrinsic_params[9]

    # p2 = intrinsic_params[12]
    # p1 = intrinsic_params[13]

    # if color_camera['Location'] == 'CALIBRATION_CameraLocationPV0':
    #     h = 1920 / 4 * 3

    #     cx *= 1920
    #     cy *= h
    #     fx *= 1920
    #     fy *= h

    #     cy -= (1920 / 4 * 3 - 1920 / 16 * 9) / 2

    # extrinsic_params = color_camera['Rt']

    # cam_R = np.array(extrinsic_params['Rotation']).reshape(3,3) 
    # cam_T = np.array(extrinsic_params['Translation'])

    # cam_intrinsics = np.array([[fx, 0., cx, 0.],
    #                            [0., fy, cy, 0.],
    #                            [0., 0., 1., 0.],
    #                            [0., 0., 0., 1.]])

    # K_rgb_distorted = [[915.8216 ,   0.     , 957.34607],
    #                   [  0.     , 915.64954, 543.62683],
    #                   [  0.     ,   0.     ,   1.     ]]

    # K_rgb_distorted = [[3.658652832031249886e+02, 0.00, 2.213334716796874773e+02], 
    #                   [0.00, 3.659283691406250227e+02, 3.832847412109375114e+02 ],
    #                   [0.00, 0.00, 1.00]]

    cam_intrinsics = np.eye(4)
    cam_intrinsics[:3,:3] = cam_intrinsic # K_rgb_distorted
    cam_intrinsics = cam_intrinsics 

    # import pdb; pdb.set_trace()
    # cam_R = torch.tensor(cam_R, device=device).expand(1,-1,-1)
    # cam_T = torch.tensor(cam_T, device=device).expand(1,-1)    
    center = torch.tensor(cam_intrinsics[0:2,2], device=device).unsqueeze(0)

    cam = PerspectiveCamera(focal_length_x = torch.tensor((cam_intrinsics[0,0].astype(np.float32))), 
                            focal_length_y = torch.tensor(cam_intrinsics[1,1].astype(np.float32)), 
                            center=center,
                            translation=T.to(device)).to(device)
    
    half_max_length = max(cam_intrinsics[0:2,2])
    principal_point = [-(cam_intrinsics[0,2]-img_size[1]/2.)/(img_size[1]/2.), \
                       -(cam_intrinsics[1,2]-img_size[0]/2.)/(img_size[0]/2.)]    
    principal_point = torch.tensor(principal_point, device=device).unsqueeze(0)
    render_image_size = max(img_size[0], img_size[1])
    f = torch.tensor([(cam_intrinsics[0,0]/half_max_length).astype(np.float32), \
                      (cam_intrinsics[1,1]/half_max_length).astype(np.float32)]).unsqueeze(0)
    renderer = SMPLRenderer(batch_size, render_image_size, f, principal_point, t=T.to(device), gender=gender).to(device)
    
    return cam, renderer

if __name__ == "__main__":
    cam_renderer_init()