import cv2
import numpy as np
import glob
import os

subject = 'outdoors_fencing_01'
seq = 'outdoors_fencing_01_masked_wo_disp_freeze_20_every_20_opt_pose' # f'{subject}_wo_disp_freeze_20_every_20_opt_pose'
result_dir = f'/home/chen/RGB-PINA/code/outputs/ThreeDPW/{seq}'
data_dir = f'/home/chen/RGB-PINA/data/{subject}'


image_paths = sorted(glob.glob(os.path.join(data_dir, 'image/*.png')))
mask_paths = sorted(glob.glob(os.path.join('/home/chen/RGB-PINA/data/outdoors_fencing_01_masked', 'mask/*.png')))
normal_paths = sorted(glob.glob(os.path.join(result_dir, 'test_normal/*.png')))

start_frame = 0
end_frame = len(image_paths)

save_dir = os.path.join(result_dir, 'test_overlay_normal')

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for idx, image_path in enumerate(image_paths[start_frame:end_frame]):
    mask_path = mask_paths[idx]
    normal_path = normal_paths[idx]
    
    image = cv2.imread(image_path)
    mask = (cv2.imread(mask_path)[:,:,-1] > 127)[:, :, np.newaxis]
    normal = cv2.imread(normal_path)
    # blueC = normal[:, :, 0]
    # greenC = normal[:, :, 1]
    # redC = normal[:, :, 2]
    # mask = ((blueC == 127) & (greenC == 127) & (redC == 127))



    overlay = (normal * mask + image * (1 - mask)).astype(np.uint8)
    # image[np.nonzero(1-mask)[0], np.nonzero(1-mask)[1]] = [0, 0, 0]
    # normal[np.nonzero(mask)[0], np.nonzero(mask)[1]] = [0, 0, 0]
    # overlay = image + normal
    cv2.imwrite(os.path.join(save_dir, f'{idx:04d}.png'), overlay)


