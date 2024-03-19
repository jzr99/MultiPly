import numpy as np
import cv2
import os
import glob
from tqdm import tqdm

seq = "pair18_basketball18_vitpose_4"
DATA_DIR = f"/cluster/project/infk/hilliges/jiangze/V2A/RGB-PINA/data/{seq}"
OUTPUT_DIR = DATA_DIR + "/edge"
os.makedirs(f"{DATA_DIR}/edge", exist_ok=True)

img_list = sorted(glob.glob(f"{DATA_DIR}/image/*.png"))
kernel = np.ones((5, 5), np.uint8)
for img_path in tqdm(img_list):
    # Read the original image
    img = cv2.imread(img_path,flags=0)
    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img, (3,3), sigmaX=0, sigmaY=0)
    # Canny Edge Detection
    edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=150)
    edges = cv2.dilate(edges, kernel)
    cv2.imwrite(f"{OUTPUT_DIR}/{os.path.basename(img_path)}", edges)
    # break
    # Display Canny Edge Detection Image
    # cv2.imshow('Canny Edge Detection', edges)
# cv2.waitKey(0)
