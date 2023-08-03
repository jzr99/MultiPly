import cv2
import glob
import numpy as np
import os
Root = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/party_dance_depth_samGT_personid_triplane_surface/test_rendering"
# Root = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/cycle_dance_sam/test_rendering"
# Root = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/Hi4D_pair19_piggyback19_loop_temporal_SAM_auto/test_normal"
# Root = "/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/code/outputs/Hi4D/courtyard_shakeHands_00_loop_temporal_SAM/test_rendering"
image_path_all = sorted(glob.glob(f"{Root}/-1/*.png"))
image_path_0 = sorted(glob.glob(f"{Root}/0/*.png"))
image_path_1 = sorted(glob.glob(f"{Root}/1/*.png"))

os.makedirs(f"{Root}/all", exist_ok=True)
for idx,(all, i, j) in enumerate(zip(image_path_all, image_path_0, image_path_1)):
    all_image=cv2.imread(all)
    i_image = cv2.imread(i)
    j_image = cv2.imread(j)
    w=i_image.shape[0]
    i_j_image = np.concatenate([i_image[w//2:],j_image[w//2:]],axis=0)
    result = np.concatenate([all_image, i_j_image],axis=1)
    # result = np.concatenate([all_image, i_image, j_image], axis=1)
    cv2.imwrite(f"{Root}/all/{int(idx):04d}.png", result)


# mask
# Root_1 = "/Users/jiangzeren/PycharmProjects/V2A/RGB-PINA/code/outputs/Hi4D/courtyard_shakeHands_00_loop/V2A_mask/V2A_1"
# Root_2 = "/Users/jiangzeren/PycharmProjects/V2A/RGB-PINA/code/outputs/Hi4D/courtyard_shakeHands_00_loop/V2A_mask/SAM_1"
# Root_3 = "/Users/jiangzeren/PycharmProjects/V2A/RGB-PINA/code/outputs/Hi4D/courtyard_shakeHands_00_loop/V2A_mask"
# image_path_all = sorted(glob.glob(f"{Root_1}/*.png"))
# image_path_0 = sorted(glob.glob(f"{Root_2}/*.png"))
# # image_path_1 = sorted(glob.glob(f"{Root_3}/1/*.png"))
#
# os.makedirs(f"{Root_3}/SAM_V2A_1", exist_ok=True)
# for idx,(all, i) in enumerate(zip(image_path_all, image_path_0)):
#     all_image=cv2.imread(all)
#     i_image = cv2.imread(i)
#     # j_image = cv2.imread(j)
#     w=i_image.shape[0]
#     # i_j_image = np.concatenate([i_image[w//2:],j_image[w//2:]],axis=0)
#     # result = np.concatenate([all_image, i_j_image],axis=1)
#     result = np.concatenate([all_image, i_image], axis=1)
#     cv2.imwrite(f"{Root_3}/SAM_V2A_1/{int(idx):04d}.png", result)