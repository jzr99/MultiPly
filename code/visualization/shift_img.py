import cv2
import numpy as np
import os
from tqdm import trange
actor = 'cg_1'
seq = 'cg_1_2'

start_frame = 13
recon_num_frames = 60
rotate_frame = 219
count = 0
shift_output_dir = '/home/chen/disk2/Neural_Avatar_eval/kinect_vis_result/%s/%s/shift_imgs' % (actor, seq)
for frame in trange(start_frame, start_frame + recon_num_frames):
    rendered_img = cv2.imread('/home/chen/disk2/Neural_Avatar_eval/kinect_vis_result/%s/%s/hardphong/%04d.png' % (actor, seq, frame)) 

    input_img = cv2.imread('/home/chen/disk2/kinect_capture_results/%s_src/%s/image/%04d.png' % (actor, seq, frame)) 
    depth_normal = cv2.imread('/home/chen/disk2/kinect_capture_results/%s_src/%s/depth_normal/%04d.png' % (actor, seq, frame)) 
    
    blank_mask = np.ones(input_img.shape, np.uint8)
    blank_mask[:, 1300:] = 0
    blank_mask[:, :700] = 0
    input_img *= blank_mask

    depth_normal[(input_img == 0)] = 255
    input_img[(input_img == 0)] = 255
    
    # cv2.imwrite('/home/chen/Desktop/blank.png', input_img)
    # import pdb
    # pdb.set_trace() 
    cv2.imwrite('/home/chen/disk2/Neural_Avatar_eval/kinect_vis_result/cg_1/cg_1_2/cropped_depth_normal/%04d.png' % count, depth_normal)
    valid_mask = (rendered_img[:,:,-1] > 0)[:, :, np.newaxis]
    output_img = (rendered_img * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
    cv2.imwrite(os.path.join(shift_output_dir, '%04d.png' % count), output_img)
    count += 1


if not os.path.exists(shift_output_dir):
    os.makedirs(shift_output_dir)

horizontal_shift = 550
for i in trange(1, horizontal_shift, 20):
    rendered_img = cv2.imread('/home/chen/disk2/Neural_Avatar_eval/kinect_vis_result/%s/%s/hardphong/%04d.png' % (actor, seq, frame)) 
    input_img = cv2.imread('/home/chen/disk2/kinect_capture_results/%s_src/%s/image/%04d.png' % (actor, seq, frame)) 
    depth_normal = cv2.imread('/home/chen/disk2/kinect_capture_results/%s_src/%s/depth_normal/%04d.png' % (actor, seq, frame)) 

    blank_mask = np.ones(input_img.shape, np.uint8)
    blank_mask[:, 1300:] = 0
    blank_mask[:, :700] = 0
    input_img *= blank_mask

    depth_normal[(input_img == 0)] = 255
    input_img[(input_img == 0)] = 255
    

    cv2.imwrite('/home/chen/disk2/Neural_Avatar_eval/kinect_vis_result/cg_1/cg_1_2/cropped_depth_normal/%04d.png' % count, depth_normal)

    shifted_img = np.zeros_like(rendered_img)

    shifted_img[:, i:, :] = rendered_img[:, :-i, :]
    valid_mask = (shifted_img[:,:,-1] > 0)[:, :, np.newaxis]
    output_img = (shifted_img * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
    cv2.imwrite(os.path.join(shift_output_dir, '%04d.png' % count), output_img)
    count += 1

for frame in trange(start_frame + recon_num_frames, rotate_frame+1):
    rendered_img = cv2.imread('/home/chen/disk2/Neural_Avatar_eval/kinect_vis_result/%s/%s/hardphong/%04d.png' % (actor, seq, frame))
    input_img = cv2.imread('/home/chen/disk2/kinect_capture_results/%s_src/%s/image/%04d.png' % (actor, seq, frame)) 
    depth_normal = cv2.imread('/home/chen/disk2/kinect_capture_results/%s_src/%s/depth_normal/%04d.png' % (actor, seq, frame)) 
    
    blank_mask = np.ones(input_img.shape, np.uint8)
    blank_mask[:, 1300:] = 0
    blank_mask[:, :700] = 0
    input_img *= blank_mask

    depth_normal[(input_img == 0)] = 255
    input_img[(input_img == 0)] = 255
    

    cv2.imwrite('/home/chen/disk2/Neural_Avatar_eval/kinect_vis_result/cg_1/cg_1_2/cropped_depth_normal/%04d.png' % count, depth_normal)

    shifted_img[:, horizontal_shift:, :] = rendered_img[:, :-horizontal_shift, :]
    valid_mask = (shifted_img[:,:,-1] > 0)[:, :, np.newaxis]
    output_img = (shifted_img * valid_mask + input_img * (1 - valid_mask)).astype(np.uint8)
    cv2.imwrite(os.path.join(shift_output_dir, '%04d.png' % count), output_img)
    count += 1

