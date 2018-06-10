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


img1 = cv2.imread("./data/box.png", 0)
img2 = cv2.imread("./data/box_in_scene.png", 0)
#TODO combine to single function
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)
print('img1 - %d features, img2 - %d features' % (len(kp1), len(kp2)))
H, vis_ = match_and_draw('find_obj', img1=img1, img2=img2, desc1=desc1, desc2=desc2, kp1=kp1, kp2=kp2)


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
    # cv2.imshow(src_proj_name, orig_src)
    _, img1 = cap.read(0)
    stored_capture = copy.copy(img1)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # img1 = cv2.equalizeHist(img1)
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

cv2.destroyAllWindows()
# stored_capture_name = "last_stored.png"
# cv2.imwrite(stored_capture_name, stored_capture)

""" 
now in "last_stored.png" we have our image to improve further
using homography matrix we'll carve from it part which corresponds to our original image
"""


#
#
while True:

    cv2.imshow("difference", stored_capture)

    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break
cv2.destroyAllWindows()

print (H)
print (np.linalg.inv(H))
carved_raw = cv2.warpPerspective(stored_capture, np.linalg.inv(H), dsize=orig_src.shape[1::-1])


from picture_descent import subtract_mean_clr

loss = np.mean(cv2.absdiff(orig_src, carved_raw))
best_loss = loss
# (orig, res, alpha, sim_treashhold
while best_loss > 0:
    best_src = copy.copy(orig_src)
    new_src = subtract_mean_clr (orig=best_src, res=carved_raw, alpha=0.5,  sim_treashhold=5)
    loss = np.mean(cv2.absdiff(new_src, carved_raw))
    if loss <= best_loss:
        best_src = new_src
    print("LOSS [{}] BEST [{}]".format(loss, best_loss))


#
# diff1 = orig_src - carved_raw
#
# cv2.imwrite("carved.png", carved_raw)




# difference_main_colour = np.median(diff1, axis=[0,1])
# dif_col = np.zeros(orig_src.shape, np.uint8)
# dif_col[:] = [difference_main_colour[0], difference_main_colour[1], difference_main_colour[2]]
#
# numpy_horizontal1 = np.hstack((orig_src, carved_raw, diff1))
#
# new_src = orig_src - (dif_col // 5)

# difference_mask = #find regions where difference near moda

# numpy_horizontal2 = np.hstack((diff1, dif_col, new_src ))



# print(dif_col.shape)

def optimise (original, resulting):
    new_original = copy.copy(original)
    best_original = copy.copy(new_original)

    loss = np.mean(cv2.absdiff(original, resulting))
    best_loss = loss

    difference_matrix= cv2.subtract(original, resulting)
    best_difference_matrix = copy.copy(difference_matrix)
    difference_matrix = original - resulting

    print("LOSS: [{}]".format(loss))
    # new_original =
    return loss, new_original



while True:

    cv2.imshow("difference", numpy_horizontal1)
    cv2.imshow("difference_colours", numpy_horizontal2)
    cv2.imshow("improved", carved_raw + dif_col)
    loss = np.mean(diff1)
    print("LOSS1: [{}]".format(loss))

    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break
cv2.destroyAllWindows()


while True:
    cv2.imshow("best", new_src)
    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break
cv2.destroyAllWindows()

#
# print(diff1.shape)
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
#
# while True:
#
#
#     _, webcam_input_frame = cap.read()
#     cv2.imshow(src_proj_name, orig_src)
#     cv2.imshow(webcam_input_name, webcam_input_frame)
#
#
#     # fps = cap.get(cv2.CAP_PROP_FPS)
#     # print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
#
#
#     ch = cv2.waitKey(1)
#     if ch == 27:  # ESC to out
#         break
#
# cv2.destroyAllWindows()

