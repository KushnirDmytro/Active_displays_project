import cv2
import numpy as np


# orig = cv2.imread("./data/baboon.jpg")
# carved = cv2.imread("./carved.png")


def subtract_mean_clr(orig, res, alpha, sim_treashhold):
    diff = orig - res
    difference_main_colour = np.median(diff, axis=[0, 1])
    dif_col = np.zeros(orig.shape, np.uint8)
    dif_col[:] = [difference_main_colour[0], difference_main_colour[1], difference_main_colour[2]]
    dist_from_mean = cv2.absdiff(diff, dif_col)
    dist_from_mean = cv2.cvtColor(dist_from_mean, cv2.COLOR_BGR2GRAY)
    mask = np.zeros(orig.shape, np.uint8)
    mask[dist_from_mean <= sim_treashhold] = [255, 255, 255]
    diff_apply = np.bitwise_and(mask, np.array(diff // (1/alpha), dtype=np.uint8 ) )
    new_best = orig - diff_apply
    return new_best

#
# diff = orig - carved
#
# difference_main_colour = np.median(diff, axis=[0,1])
# dif_col = np.zeros(orig.shape, np.uint8)
# dif_col[:] = [difference_main_colour[0], difference_main_colour[1], difference_main_colour[2]]
#
# dist_from_mean = cv2.absdiff(diff, dif_col)
#
# # print(dist_from_mean.shape)
#
# dist_from_mean = cv2.cvtColor(dist_from_mean, cv2.COLOR_BGR2GRAY)
#
# mask = np.zeros(orig.shape, np.uint8)
#
#
# mask[dist_from_mean <= 5 ] =  [255, 255, 255]
#
# print(mask.shape)
#
# diff_apply = np.bitwise_and(mask, diff)
#
#
# horis = np.hstack( (orig, carved, diff) )
# horis2 = np.hstack( (orig,  orig - diff_apply) )
#
#
#
# while True:
#     cv2.imshow("or", horis)
#     cv2.imshow("dist", dist_from_mean)
#     cv2.imshow("mask", mask)
#     cv2.imshow("apply_diff",  diff_apply)
#     cv2.imshow("res",  horis2)
#     # cv2.imshow("proj", carved)
#     #
#     ch = cv2.waitKey(1)
#     if ch == 27:  # ESC to out
#         break
# cv2.destroyAllWindows()
# # difference_mask = #find regions where difference near moda