import numpy as np
import cv2
import matplotlib.pyplot as plt

def nothing():
    pass

img1 = cv2.imread('./data/ghosty_chess.png',0)
img2 = cv2.imread('./data/chessboard.png',0)

img2 = cv2.resize(img2, (400, 400) )

img1 = cv2.equalizeHist(img1)

# cv2.createTrackbar('R','image',0,255,nothing)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(img1,None)
kp2, des2 = orb.detectAndCompute(img2,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

matches = bf.match(des1,des2)
matches = sorted(matches, key = lambda x:x.distance)

img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10],None, flags=2)

plt.imshow(img3)
plt.show()