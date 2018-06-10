import cv2
import numpy as np


# orig = cv2.imread("./data/baboon.jpg")
# carved = cv2.imread("./carved.png")

import copy

def subtract_mean_clr(orig, carved, alpha, sim_treashhold):
    prev_mask_sum = -1
    while True:
        diff = orig - carved
        difference_main_colour = np.median(diff, axis=[0, 1])
        dif_col = np.zeros(orig.shape, np.uint8)
        dif_col[:] = [difference_main_colour[0], difference_main_colour[1], difference_main_colour[2]]



        dist_from_mean = cv2.absdiff(diff, dif_col)
        # print(dist_from_mean.shape)
        dist_from_mean = cv2.cvtColor(dist_from_mean, cv2.COLOR_BGR2GRAY)

        or2 = copy.copy(orig)
        inverted_colour = np.bitwise_not(dif_col, or2)
        # cv2.imshow("experiment", res)

        mask = np.zeros(orig.shape, np.uint8)

        mask[dist_from_mean <= sim_treashhold] = [255, 255, 255]

        print(mask.shape)

        diff_apply = np.bitwise_and(mask, inverted_colour) # < =========
        diff_apply = np.array(diff_apply * 1, dtype=np.uint8)

        orig = np.bitwise_or(orig,diff_apply)

        sum_mask = np.sum(mask)
        if sum_mask <= 0 or prev_mask_sum == sum_mask:
            break
        prev_mask_sum = sum_mask
    return orig
