import cv2 as cv
# import scipyplot

def detectEyes(frame,
               faceCascade,
               eyeCascade):


  grayscale = cv.cv2.cvtColor(frame, cv.cv2.COLOR_BGR2GRAY) #; // convert image to grayscale
  grayscale = cv.cv2.equalizeHist(grayscale, grayscale) #; // enhance image contrast

  #TODO change to MOSSE_tracking  when face is detected (if it will deal with computing1)
  #present in OpenCV
# http://www.cs.colostate.edu/~vision/publications/bolme_cvpr10.pdf


#TODO tune parameters for better face/eyes detection
  faces = faceCascade.detectMultiScale(grayscale, scaleFactor=1.3, minNeighbors=2, minSize=(50, 50),
                                   flags=cv.CASCADE_SCALE_IMAGE)
  for face in faces:
      # print(face)
      (face_x, face_y, face_w, face_h) = face
      face_caption = grayscale[face_y:face_y + face_h, face_x:face_x + face_w]
      eyes = eyeCascade.detectMultiScale(face_caption, scaleFactor=1.3, minNeighbors=2, minSize=(20, 20),
                                   flags=cv.CASCADE_SCALE_IMAGE)

      eyes_centers = []
      for eye in eyes:
          (e_x, e_y, e_w, e_h) = eye
          print (eye)
          eyebox_corner1 =(face_x+ e_x,face_y+ e_y)
          eyebox_corner2 =(face_x+ e_x + e_w, face_y+e_y + e_h)

          frame = \
              cv.rectangle(
                  img = frame,
                  pt1 = eyebox_corner1,
                  pt2 = eyebox_corner2,
                  color=(0, 255, 0),
                  thickness=2)
          center_x = (eyebox_corner1[0] + eyebox_corner2[0]) // 2
          center_y = (eyebox_corner1[1] + eyebox_corner2[1]) // 2
          eyes_centers.append(  (center_x, center_y)  )
          if (len(eyes_centers)==2): #TODO check other cases, find THAT pair of eyes
              sight_senter_x = (eyes_centers[0][0] + eyes_centers[1][0])//2
              sight_senter_y = (eyes_centers[0][1] + eyes_centers[1][1]) // 2
              frame = \
                  cv.circle(
                      img=frame,
                      center=(sight_senter_x,sight_senter_y),
                      radius=3,
                      color=(0, 0, 255),
                      thickness=2)

      cv.imshow(faceName, face_caption)

      frame = cv.rectangle(frame, (face_x, face_y), (face_x+face_w, face_y+face_h), color=(255, 0, 0), thickness=2)






winName = "WebCam"
faceName = "Face_detected"
active_picture_name = "for_now_not_so_active_pic"

try:
    cap = cv.VideoCapture(0)
    cap.set(cv.cv2.CAP_PROP_FPS, 60.0)
    # TODO get detailed infro how to increase framerate and quality, now control is poor
    #TODO make multithreaded video processing

    cap.set(cv.cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.cv2.CAP_PROP_FRAME_HEIGHT, 480)
except ValueError:
    if (not cap.isOpened()):
        print("Failed to init camera")
        exit(-1)



cv.namedWindow(winName)
cv.namedWindow(faceName)
cv.namedWindow(active_picture_name)

eyeCascade = cv.CascadeClassifier()
faceCascade = cv.CascadeClassifier()

try :
    faceCascade.load("./haarcascades/haarcascade_frontalface_alt.xml")
except ValueError:
    print("Could not load face detector." )
    exit(-1)

try:
    eyeCascade.load("./haarcascades/haarcascade_eye_tree_eyeglasses.xml")
    #todo customise 'eyeglasses' mode
except ValueError:
    print("Could not load eye detector.")
    exit(-1)

sample = "./baboon.jpg"
img = cv.imread(sample)
# print(type(img))

while True:


    _, frame = cap.read()
    detectEyes(frame, faceCascade, eyeCascade)
    cv.imshow(winName, frame)

    cv.imshow(active_picture_name, img)

    (major_ver, minor_ver, subminor_ver) = (cv.__version__).split('.')

    if int(major_ver) < 3:
        pass
        fps = cap.get(cv.cv2.CV_CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    else:
        fps = cap.get(cv.cv2.CAP_PROP_FPS)
        print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))


    ch = cv.waitKey(1)
    if ch == 27:  # ESC to out
        break




cv.destroyAllWindows()
# ========================================================================================================
