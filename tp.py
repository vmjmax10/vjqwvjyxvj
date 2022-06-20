# from yolox.core import Trainer, launch
# from yolox.exp import get_exp
# from yolox.utils import configure_nccl, configure_omp, get_num_devices


import cv2, numpy as np


def augment_hsv(img, hgain=0.015, sgain=0.7, vgain=0.4):
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    dtype = img.dtype  # uint8

    x = np.arange(0, 256, dtype=np.int16)
    lut_hue = ((x * r[0]) % 180).astype(dtype)
    lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    img_hsv = cv2.merge(
        (cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))
    ).astype(dtype)
    cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed


img = cv2.imread("1.jpg")

# img1 = img.copy()
# augment_hsv(img1)

# img2 = img.copy()
# augment_hsv(img2)

# img3 = img.copy()
# augment_hsv(img3)

img1 = img[:, ::-1]

cv2.namedWindow("disp", cv2.WINDOW_NORMAL)
cv2.imshow("disp", cv2.hconcat([img, img1]))
cv2.waitKey(0)
