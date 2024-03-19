import cv2
import numpy as np
import glob
import os
from tqdm import tqdm

# subject = 'ma_wechat1'
# seq = f'{subject}'
# result_dir = f'/home/chen/release_tmp/RGB-PINA/outputs/Video/{seq}'
result_dir = '/media/ubuntu/hdd/V2A_output/v2as_share'
data_dir = f'/media/ubuntu/hdd/RGB-PINA/data/dance4'

folder_list = sorted(glob.glob(os.path.join(result_dir, 'dance4_low_*')))
number_person = len(folder_list)
image_paths = sorted(glob.glob(os.path.join(data_dir, 'image/*.png')))

mask_path_folder = []
normal_path_folder = []
depth_path_folder = []
for p in range(number_person):
    mask_paths = sorted(glob.glob(os.path.join(folder_list[p], 'test_mask/*.png')))
    normal_paths = sorted(glob.glob(os.path.join(folder_list[p], 'test_normal/*.png')))
    depth_paths = sorted(glob.glob(os.path.join(folder_list[p], 'test_depth/*.npy')))
    mask_path_folder.append(mask_paths)
    normal_path_folder.append(normal_paths)
    depth_path_folder.append(depth_paths)
    

start_frame = 0
end_frame = len(image_paths)
# end_frame = 290

save_dir = os.path.join(folder_list[0], 'test_overlay_normal')
save_dir_noimage = os.path.join(folder_list[0], 'test_overlay_normal_noimage')
os.makedirs(save_dir_noimage, exist_ok=True)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for idx, image_path in enumerate(tqdm(image_paths[start_frame:end_frame])):
    image = cv2.imread(image_path)
    mask_np = []
    normal_np = []
    depth_np = []
    for p in range(number_person):
        mask_paths = mask_path_folder[p]
        normal_paths = normal_path_folder[p]
        depth_paths = depth_path_folder[p]
        mask_path = mask_paths[idx]
        normal_path = normal_paths[idx]
        depth_path = depth_paths[idx]
        mask=cv2.imread(mask_path)
        dial_kernel = np.ones((3, 3),np.uint8)
        mask = cv2.erode(mask, dial_kernel)
        mask = (mask[:,:,-1] > 127)[:, :, np.newaxis]
        normal = cv2.imread(normal_path)
        depth = np.load(depth_path)
        depth = depth * mask
        # important !!!!!!!!!!!!!!!!!!!!!!! depth map is gradually reduce to 0 from the body to the background, so the edge of the normal may have a unreasonable small depth value, here I use a threshold to filter out that depth value to 0
        # depth[depth<2.3] = 0
        mask_np.append(mask)
        normal_np.append(normal)
        depth_np.append(depth)
    mask_np = np.concatenate(mask_np, 2)
    # stack normal np
    normal_np = np.stack(normal_np, 0)
    depth_np = np.concatenate(depth_np, 2)
    # union the mask
    mask = np.sum(mask_np, 2) > 0
    mask = mask[:, :, np.newaxis]
    # select the normal image based on the depth
    depth_np[depth_np==0] = 999
    # select the index of the minimum depth
    min_idx = np.argmin(depth_np, 2)
    w, h = min_idx.shape
    # select the corresponding normal_np based on the min_idx
    normal_np = normal_np.reshape(normal_np.shape[0],w*h, 3)
    normal_selected = normal_np[min_idx.reshape(-1), np.arange(w*h),:]
    normal_selected = normal_selected.reshape(w,h,3)

    overlay = (normal_selected * mask + image * (1 - mask)).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'{idx:04d}.png'), overlay)
    cv2.imwrite(os.path.join(save_dir_noimage, f'{idx:04d}.png'), normal_selected)