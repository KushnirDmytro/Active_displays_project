import cv2

winName = "WebCam"
active_picture_name = "for_now_not_so_active_pic"

try:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 60.0)
    # TODO get detailed infro how to increase framerate and quality, now control is poor
    #TODO make multithreaded video processing

    # (7680, 4320)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 780)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
except ValueError:
    if (not cap.isOpened()):
        print("Failed to init camera")
        exit(-1)




cv2.namedWindow(winName)
# cv.namedWindow(active_picture_name)

# eyeCascade = cv.CascadeClassifier()
# faceCascade = cv.CascadeClassifier()

# try :
#     faceCascade.load("./haarcascades/haarcascade_frontalface_alt.xml")
# except ValueError:
#     print("Could not load face detector." )
#     exit(-1)
#
# try:
#     eyeCascade.load("./haarcascades/haarcascade_eye_tree_eyeglasses.xml")
#     #todo customise 'eyeglasses' mode
# except ValueError:
#     print("Could not load eye detector.")
#     exit(-1)

# sample = "./baboon.jpg"
# img = cv.imread(sample)
# print(type(img))

while True:


    _, frame = cap.read()
    # detectEyes(frame, faceCascade, eyeCascade)
    cv2.imshow(winName, frame)

    # cv.imshow(active_picture_name, img)

    (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')

    if int(major_ver) < 3:
        pass
        fps = cap.get(cv2.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break




cv2.destroyAllWindows()