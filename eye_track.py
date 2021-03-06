import cv2 as cv
from traingulation_of_position import warp_the_image_in_3d
import numpy as np
import math
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
      frame = cv.rectangle(frame, (face_x, face_y), (face_x + face_w, face_y + face_h), color=(255, 0, 0), thickness=2)
      eyes_centers = []
      for eye in eyes:
          (e_x, e_y, e_w, e_h) = eye
          # print (eye)
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
              vector_to_sight = [sight_senter_x, sight_senter_y]



              return vector_to_sight

      cv.imshow(faceName, face_caption)


print("1")




winName = "WebCam"
faceName = "Face_detected"
active_picture_name = "for_now_not_so_active_pic"


# camera_width = 800
# camera_height = 600


try:
    cap = cv.VideoCapture(0)
    cap.set(cv.cv2.CAP_PROP_FPS, 60.0)
    # TODO get detailed infro how to increase framerate and quality, now control is poor
    #TODO make multithreaded video processing

    # cap.set(cv.cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    # cap.set(cv.cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

except ValueError:
    if (not cap.isOpened()):
        print("Failed to init camera")
        exit(-1)


cv.namedWindow(winName)
cv.namedWindow(faceName)
cv.namedWindow(active_picture_name)

eyeCascade = cv.CascadeClassifier()
faceCascade = cv.CascadeClassifier()


#LETITBE
distance_to_face = 1000


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

# sample = "./baboon.jpg"
# img = cv.imread(sample)
# print(type(img))


_, frame = cap.read()

frame_X, frame_Y, depth = frame.shape
camera_center = (frame_X//2, frame_Y//2)
print (camera_center)

# new_img = img
# img_X, img_Y, _ = img.shape
# img_center = (img_X//2, img_Y //2)
while True:


    _, frame = cap.read()
    camera_sight_pos = detectEyes(frame, faceCascade, eyeCascade)

    if (camera_sight_pos):
        print ("CSP:{}".format(camera_sight_pos))
        face_shift_from_center_of_camera = [camera_sight_pos[0] - camera_center[0], camera_sight_pos[1] - camera_center[1]]
        print("CenterShift:{}".format(face_shift_from_center_of_camera))
        camera_shift_relative = (face_shift_from_center_of_camera[0]/frame_X, face_shift_from_center_of_camera[1]/frame_Y)

        # projection_on_standart_plane = (img_X*camera_shift_relative[0] ,  img_Y * camera_shift_relative[1])

        # z_coord = math.sqrt (distance_to_face*distance_to_face - projection_on_standart_plane[0]*projection_on_standart_plane[0] - \
        #           projection_on_standart_plane[1]*projection_on_standart_plane[1])
        
        # ptsSrc = np.array([
        #     [img_X, 0, 0],
        #     [img_X, img_Y, 0],
        #     [img_center[0], img_center[1], 0], #the same
        #     [projection_on_standart_plane[0], projection_on_standart_plane[1], z_coord] #calculated
        # ])
        #
        #
        # ptsDest = ([
        #     # [img_X, 0, 0],
        #     # [img_X, img_Y, 0],
        #     [img_center[0], img_center[1], 0],
        #     [img_center[0], img_center[1], distance_to_face]
        # ])

        # new_img = warp_the_image_in_3d(imgX, )

    frame = \
        cv.circle(
            img=frame,
            center=(camera_center[1], camera_center[0]),
            radius=3,
            color=(255, 0, 255),
            thickness=2)

    cv.imshow(winName, frame)
    #
    #
    #
    # cv.imshow(active_picture_name, img)

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
