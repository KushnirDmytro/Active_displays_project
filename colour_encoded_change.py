import cv2

import numpy as np


orig = cv2.imread('./data/baboon.jpg',1)
kernel = np.ones((10, 10), np.float32) / 100
orig = cv2.filter2D(orig,-1,kernel)
distort = cv2.imread('./data/encoded1.png',1)



orig_hsv  = cv2.cvtColor(orig, cv2.COLOR_BGR2YCrCb)



dist_hsv  = cv2.cvtColor(distort, cv2.COLOR_BGR2YCrCb)
gray_hsv = cv2.cvtColor(distort, cv2.COLOR_BGR2GRAY)
print (orig_hsv.shape)

# print (type (orig))
diff1 = cv2.subtract( dist_hsv, orig_hsv)
# diff2 = distort - orig

# print(diff1.shape)
# col = np.median(diff1, axis=[0,1])
#
# dif_col = np.zeros(orig.shape, np.uint8)
# dif_col[:] = [col[0], col[1], col[2]]
#
#
# loss = np.mean(diff1)
# print("LOSS1: [{}]".format(loss))
#
# distort1 = distort + dif_col
# diff2 = orig - distort1
# col2 = np.median(diff2, axis=[0,1])
# loss2 = np.mean(diff2)
# print("LOSS2: [{}]".format(loss2) )
# dif_col2 = np.zeros(orig.shape, np.uint8)
# dif_col2[:] = [col2[0], col2[1], col2[2]]

#
print(orig_hsv[:, :, 0].shape)
while True:

    # cv2.imshow("or", orig)
    # cv2.imshow("ch1", orig[:,:,0])
    # cv2.imshow("ch2", orig[:,:,1])
    # cv2.imshow("ch3", orig[:,:,2])

    cv2.imshow("orig", orig)
    cv2.imshow("orig_HSV", orig_hsv )
    cv2.imshow("diff", diff1 )
    cv2.imshow("diff_1", diff1[:, :, 0] )
    cv2.imshow("diff_2", diff1[:, :, 1] )
    cv2.imshow("diff_3", diff1[:, :, 2] )
    cv2.imshow("orig_gray", gray_hsv )

    # cv2.imshow("dist", distort)
    # cv2.imshow("dist_HSV", dist_hsv)
#     cv2.imshow("diff", diff1)
#     cv2.imshow("diff_col", dif_col)
#     cv2.imshow("diff2", diff2)
#     cv2.imshow("disstort1", distort1)
#     cv2.imshow("diff_col2", dif_col2)
#     # cv2.imshow("diff3", diff3)
#     # cv2.imshow("disstort2", distort2)
#     # cv2.imshow("diff_col3", dif_col3)

    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break

