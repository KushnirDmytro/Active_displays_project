import cv2 as cv
import numpy as np

H = np.identity(3)
nOctaveLayers_ =  1
contrastThreshold_  = 1
edgeThreshold_ = 1
sigma_ = 1
n_features_ = 1

cmp_value = 5.0

def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    kp_pairs = zip(mkp1, mkp2)
    return p1, p2, list(kp_pairs)

def explore_match(win, img1, img2, kp_pairs, status = None, H = None):
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    vis = np.zeros((max(h1, h2), w1+w2), np.uint8)
    vis[:h1, :w1] = img1
    vis[:h2, w1:w1+w2] = img2
    vis = cv.cvtColor(vis, cv.COLOR_GRAY2BGR)

    if H is not None:
        corners = np.float32([[0, 0], [w1, 0], [w1, h1], [0, h1]])
        corners = np.int32( cv.perspectiveTransform(corners.reshape(1, -1, 2), H).reshape(-1, 2) + (w1, 0) )
        cv.polylines(vis, [corners], True, (255, 255, 255))

    if status is None:
        status = np.ones(len(kp_pairs), np.bool_)
    p1, p2 = [], []  # python 2 / python 3 change of zip unpacking
    for kpp in kp_pairs:
        p1.append(np.int32(kpp[0].pt))
        p2.append(np.int32(np.array(kpp[1].pt) + [w1, 0]))

    green = (0, 255, 0)
    red = (0, 0, 255)
    kp_color = (51, 103, 236)
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            col = green
            cv.circle(vis, (x1, y1), 2, col, -1)
            cv.circle(vis, (x2, y2), 2, col, -1)
        else:
            col = red
            r = 2
            thickness = 3
            cv.line(vis, (x1-r, y1-r), (x1+r, y1+r), col, thickness)
            cv.line(vis, (x1-r, y1+r), (x1+r, y1-r), col, thickness)
            cv.line(vis, (x2-r, y2-r), (x2+r, y2+r), col, thickness)
            cv.line(vis, (x2-r, y2+r), (x2+r, y2-r), col, thickness)
    vis0 = vis.copy()
    for (x1, y1), (x2, y2), inlier in zip(p1, p2, status):
        if inlier:
            cv.line(vis, (x1, y1), (x2, y2), green)

    cv.imshow(win, vis)

    def onmouse(event, x, y, flags, param):
        cur_vis = vis
        if flags & cv.EVENT_FLAG_LBUTTON:
            cur_vis = vis0.copy()
            r = 8
            m = (anorm(np.array(p1) - (x, y)) < r) | (anorm(np.array(p2) - (x, y)) < r)
            idxs = np.where(m)[0]

            kp1s, kp2s = [], []
            for i in idxs:
                (x1, y1), (x2, y2) = p1[i], p2[i]
                col = (red, green)[status[i][0]]
                cv.line(cur_vis, (x1, y1), (x2, y2), col)
                kp1, kp2 = kp_pairs[i]
                kp1s.append(kp1)
                kp2s.append(kp2)
            cur_vis = cv.drawKeypoints(cur_vis, kp1s, None, flags=4, color=kp_color)
            cur_vis[:,w1:] = cv.drawKeypoints(cur_vis[:,w1:], kp2s, None, flags=4, color=kp_color)

        cv.imshow(win, cur_vis)
    cv.setMouseCallback(win, onmouse)
    return vis

def match_and_draw(win, img1, img2, desc1, desc2, kp1, kp2, matcher):
    print('matching...')
    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  # 2
    p1, p2, kp_pairs = filter_matches(kp1, kp2, raw_matches)
    if len(p1) >= 4:
        # cmp_value = 5.0
        H, status = cv.findHomography(p1, p2, cv.RANSAC, cmp_value)
        print('%d / %d  inliers/matched' % (np.sum(status), len(status)))
    else:
        H, status = None, None
        print('%d matches found, not enough for homography estimation' % len(p1))

    vis_ = explore_match(win, img1, img2, kp_pairs, status, H)
    return H, vis_

def nothing(x):
    pass
# from __future__ import print_function

import numpy as np
# from common import anorm, getsize


FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6

import sys, getopt
from common import anorm, getsize

opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
opts = dict(opts)
# feature_name = opts.get('--feature', 'brisk')
feature_name = opts.get('--feature', 'sift')
# feature_name = opts.get('--feature', 'surf')
# feature_name = opts.get('--feature', 'orb')
# feature_name = opts.get('--feature', 'akaze')
# feature_name = 'flann'

try:
    fn1, fn2 = args
except:
    # fn1 = './data/night_openCV_logo.png'
    # fn2 = './data/opencv-logo.png'
    fn1 = './data/night_baboon.png'
    fn2 = './data/baboon.jpg'
    # img1 = cv2.imread('./ghosty_chess.png', 0)
    # img2 = cv2.imread('./chessboard.png', 0)

img1 = cv.imread(fn1, 0)
original1 = cv.imread(fn1, 1)
# clahe = cv.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
# img1 = clahe.apply(img1)
# cv.imshow("before", img1)

# img1 = cv.equalizeHist(img1)

# cv.imshow("after", img1)


img2 = cv.imread(fn2, 0)
original2 = cv.imread(fn2, 1)
# img2 = cv.resize(img2, (400, 400) )


swap = img1
img1 = img2
img2 = swap

# original = "./baboon.jpg"
# projected_raw = "./bad_img_1.png"

# orig = cv.imread(original)
# proj_raw = cv.imread(projected_raw)

win1_name = "original"
win2_name = "bad_proj"


def init_feature(name):
    chunks = name.split('-')
    chunks += ['flann']
    detector = cv.xfeatures2d.SIFT_create(
        # nfeatures = n_features_,
        # nOctaveLayers=nOctaveLayers_,
        # contrastThreshold = contrastThreshold_ ,
        # edgeThreshold =edgeThreshold_ ,
        # sigma = sigma_
    )
    norm = cv.NORM_L2

    if norm == cv.NORM_L2:
        flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    else:
        flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
    matcher = cv.FlannBasedMatcher(flann_params, {})
    return detector, matcher



detector, matcher = init_feature(feature_name)

# if img1 is None:
#     print('Failed to load fn1:', fn1)
#     sys.exit(1)
#
# if img2 is None:
#     print('Failed to load fn2:', fn2)
#     sys.exit(1)
#
# if detector is None:
#     print('unknown feature:', feature_name)
#     sys.exit(1)

print('using', feature_name)

kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)
print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))


H, vis_ = match_and_draw('find_obj', img1=img1, img2=img2, desc1=desc1, desc2=desc2,  kp1=kp1, kp2=kp2)

cv.createTrackbar('n_features','find_obj',1,100,nothing)
cv.createTrackbar('nOctaveLayers','find_obj',1,10,nothing)
cv.createTrackbar('contrastThreshold','find_obj',1,100,nothing)
cv.createTrackbar('edgeThreshold','find_obj',1,10,nothing)
cv.createTrackbar('sigma','find_obj',1,20,nothing)
cv.createTrackbar('cmp_value','find_obj',1,20,nothing)
switch = '0 : MANUAL CALIBTRATION OFF \n1 : MANUAL CALIBRATION ON'
cv.createTrackbar(switch, 'find_obj',0,1,nothing)


while True:
    break

    n_features_ = cv.getTrackbarPos('n_features','find_obj')
    nOctaveLayers_ = cv.getTrackbarPos('nOctaveLayers','find_obj')
    contrastThreshold_  = cv.getTrackbarPos('contrastThreshold','find_obj') / 100
    edgeThreshold_ = cv.getTrackbarPos('edgeThreshold','find_obj')
    sigma_ = cv.getTrackbarPos('sigma','find_obj') / 10

    print ("features: [{}] layers: [{}] contrast_tr: [{}] edge: [{}] sigma: [{}]".format(
        n_features_,nOctaveLayers_,contrastThreshold_,edgeThreshold_,sigma_
    ))

    detector, matcher = init_feature(feature_name)
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
    H, vis_ = match_and_draw('find_obj')


    ch = cv.waitKey(1)
    if ch == 27:  # ESC to out
        break
cv.destroyAllWindows()

print ("SHAPE before: [{}]".format(original1.shape))

a = cv.warpPerspective(original1, np.linalg.inv(H), dsize=img1.shape)

print ("SHAPE final: [{}]".format(a.shape))

while True:
    break
    cv.imshow('transformed', a)

    ch = cv.waitKey(1)
    if ch == 27:  # ESC to out
        break
cv.destroyAllWindows()
# cv.imwrite("./data/captured.png", a)

# static Ptr<SIFT> cv::xfeatures2d::SIFT::create	(	int 	nfeatures = 0,
# int 	nOctaveLayers = 3,
# double 	contrastThreshold = 0.04,
# double 	edgeThreshold = 10,
# double 	sigma = 1.6
# )

# while True:
#     cv2.imshow(win1_name, orig)
#     cv2.imshow(win2_name, proj_raw)
#     ch = cv2.waitKey(1)
#     if ch == 27:  # ESC to out
#         break

# cv.destroyAllWindows()