import cv2
import numpy as np
from find_correspondance_final import nothing, match_and_draw
import copy



#here are defined some usefull but long and irrelevant functions for image matching

#=======================+STAGE 1+ detect ROI =======================
"""
in this part of execution pipeline program detects region of projection on a surface.
It can be done eather in auto mode or manually adjust detector settings (as condition of usage are
 intended to be unpleasant, this tradeoff seems inevitable).
  Other way can be calibration using "changing" picture to detect altering region of picture. 
  But such approach has other tradoffs:
   1) recalibration interrupts ordinary usage process. 
   2) other movig or altering regions of captured picture can spoil the results
"""


"""
projected img
"""
src_name = "./data/baboon.jpg"
orig_src = cv2.imread(src_name,1)
best_src = copy.copy(orig_src)
src_proj_name = "src_proj"
#============================================

"""
capturing image configuration channel
"""
webcam_input_name = "WebCam"
active_picture_name = "for_now_not_so_active_pic"

try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60.0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 780)

except ValueError:
    if (not cap.isOpened()):
        print("Failed to init camera")
        exit(-1)
#=============================================


"""
init features extractor and matcher for homography matrix estimation
"""


def init_feature(args_dict):
    FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
    FLANN_INDEX_LSH =6
    if 'detect_manual' in args_dict and args_dict['detect_manual'] == 1:
        detector = cv2.xfeatures2d.SIFT_create(
            nfeatures=args_dict['nfeatures'],
            nOctaveLayers=args_dict['nOctaveLayers'],
            contrastThreshold=args_dict['contrastThreshold'],
            edgeThreshold=args_dict['edgeThreshold'],
            sigma=args_dict['sigma'],
        )
    else:
        detector = cv2.xfeatures2d.SIFT_create( )

    norm = cv2.NORM_L2

    if norm == cv2.NORM_L2:
        flann_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    else:
        flann_params = dict(algorithm=FLANN_INDEX_LSH,
                            table_number=6,  # 12
                            key_size=12,  # 20
                            multi_probe_level=1)  # 2
    matcher = cv2.FlannBasedMatcher(flann_params, {})
    return detector, matcher

detector, matcher = init_feature({})


cv2.imshow('find_obj', orig_src)
cv2.createTrackbar('n_features','find_obj',1,100,nothing)
cv2.createTrackbar('nOctaveLayers','find_obj',1,10,nothing)
cv2.createTrackbar('contrastThreshold','find_obj',1,100,nothing)
cv2.createTrackbar('edgeThreshold','find_obj',1,10,nothing)
cv2.createTrackbar('sigma','find_obj',1,20,nothing)
cv2.createTrackbar('cmp_value','find_obj',1,20,nothing)
auto_switch = '0 : MANUAL CALIBTRATION OFF \n1 : MANUAL CALIBRATION ON'
cv2.createTrackbar(auto_switch, 'find_obj',0,1,nothing)



def project_and_read(to_project):
    pass




while True:

    cv2.imshow('permanent_baboon', best_src)

    _, img1 = cap.read(0)
    stored_capture = copy.copy(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img1 = cv2.medianBlur(img1, 5)  # to improve quality of img
    img2 = orig_src
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    swap = img1
    img1 = img2
    img2 = swap

    args_dict = {}
    auto_state = cv2.getTrackbarPos(auto_switch, 'find_obj')
    if auto_state == 1:
        args_dict['detect_manual'] = 1
        args_dict['nfeatures'] = cv2.getTrackbarPos('n_features', 'find_obj')
        args_dict['nOctaveLayers'] = cv2.getTrackbarPos('nOctaveLayers', 'find_obj')
        args_dict['contrastThreshold'] = cv2.getTrackbarPos('contrastThreshold', 'find_obj') / 100
        args_dict['edgeThreshold'] = cv2.getTrackbarPos('edgeThreshold', 'find_obj')
        args_dict['sigma'] = cv2.getTrackbarPos('sigma', 'find_obj') / 10

    config = ""
    for key in args_dict:
        config += (key + " [{}] ").format(args_dict[key])
    print(config)
    detector, matcher = init_feature(args_dict)
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
    H, vis_ = match_and_draw('find_obj', img1=img1, img2=img2, desc1=desc1, desc2=desc2,  kp1=kp1, kp2=kp2)

    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break

cv2.destroyWindow('find_obj')

""" 
now in "last_stored.png" we have our image to improve further
using homography matrix we'll carve from it part which corresponds to our original image
"""


carved_raw = cv2.warpPerspective(stored_capture, np.linalg.inv(H), dsize=orig_src.shape[1::-1])


from picture_descent import subtract_mean_clr

loss = np.mean(cv2.absdiff(orig_src, carved_raw))
best_loss = loss
new_src = best_src
while True:
    _, stored_capture = cap.read(0)

    new_src = subtract_mean_clr (orig=orig_src, carved=carved_raw, alpha=0.2,  sim_treashhold=10)

    cv2.imshow('permanent_baboon', best_src)
    cv2.imshow('copy_baboon', best_src)
    cv2.imshow('new_src', new_src)
    cv2.imshow('carved', carved_raw)

    carved_raw = cv2.warpPerspective(stored_capture, np.linalg.inv(H), dsize=orig_src.shape[1::-1])

    loss = np.mean(cv2.absdiff(orig_src, carved_raw))
    cv2.imshow('orig', orig_src)

    if loss <= best_loss:
        best_src = copy.copy(new_src)
        best_loss = loss
    print("LOSS [{}] BEST [{}]".format(loss, best_loss))

    args_dict = {}

    detector, matcher = init_feature(args_dict)
    kp1, desc1 = detector.detectAndCompute(img1, None)
    kp2, desc2 = detector.detectAndCompute(img2, None)
    print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
    H, vis_ = match_and_draw('find_obj', img1=img1, img2=img2, desc1=desc1, desc2=desc2, kp1=kp1, kp2=kp2)

    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break
cv2.destroyAllWindows()







