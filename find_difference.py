import cv2

import numpy as np


orig = cv2.imread('./data/baboon.jpg',1)
distort = cv2.imread('./data/encoded1.png',1)

# print (type (orig))
diff1 = orig - distort
# diff2 = distort - orig

print(diff1.shape)
col = np.median(diff1, axis=[0,1])

dif_col = np.zeros(orig.shape, np.uint8)
dif_col[:] = [col[0], col[1], col[2]]


loss = np.mean(diff1)
print("LOSS1: [{}]".format(loss))

distort1 = distort + dif_col
diff2 = orig - distort1
col2 = np.median(diff2, axis=[0,1])
loss2 = np.mean(diff2)
print("LOSS2: [{}]".format(loss2) )
dif_col2 = np.zeros(orig.shape, np.uint8)
dif_col2[:] = [col2[0], col2[1], col2[2]]

import copy
while True:



    cv2.imshow("orig", orig)
    cv2.imshow("dist", distort)
    cv2.imshow("diff", diff1)
    cv2.imshow("diff_col", dif_col)
    cv2.imshow("diff2", diff2)
    cv2.imshow("disstort1", distort1)
    cv2.imshow("diff_col2", dif_col2)



    or2 = copy.copy(orig)
    res = np.bitwise_not(dif_col, or2)
    cv2.imshow("experiment", res)

    cv2.imshow("ex", np.hstack((orig, distort, diff1)))
    cv2.imshow("ex2", np.hstack((diff1, dif_col, res)))

    # cv2.imshow("diff3", diff3)
    # cv2.imshow("disstort2", distort2)
    # cv2.imshow("diff_col3", dif_col3)

    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break

