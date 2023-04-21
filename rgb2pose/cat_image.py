import cv2
import glob
import numpy as np

Root = "/Users/jiangzeren/PycharmProjects/V2A/RGB-PINA/code/outputs/Hi4D/pair01_hug01_cam4_all/test_normal"
image_path_all = sorted(glob.glob(f"{Root}/-1/*.png"))
image_path_0 = sorted(glob.glob(f"{Root}/0/*.png"))
image_path_1 = sorted(glob.glob(f"{Root}/1/*.png"))


for idx,(all, i, j) in enumerate(zip(image_path_all, image_path_0, image_path_1)):
    all_image=cv2.imread(all)
    i_image = cv2.imread(i)
    j_image = cv2.imread(j)
    # w=i_image.shape[0]
    # i_j_image = np.concatenate([i_image[w//2:],j_image[w//2:]],axis=0)
    # result = np.concatenate([all_image, i_j_image],axis=1)
    result = np.concatenate([all_image, i_image, j_image], axis=1)
    cv2.imwrite(f"{Root}/all/{int(idx):04d}.png", result)