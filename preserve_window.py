import cv2
"""
capturing image configuration channel
"""
webcam_input_name = "WebCam"
active_picture_name = "for_now_not_so_active_pic"

im2 = cv2.imread('encoded1.png')
im3 = cv2.imread('carved.png')

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

while True:
    cv2.imshow('dumm', im2)
    _, cap_img = cap.read(0)
    cv2.imshow("cap", cap_img)
    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break

cv2.destroyWindow('dumm')

while True:
    cv2.imshow('dumm2', im3)
    _, cap_img = cap.read(0)
    cv2.imshow("cap", cap_img)
    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break

cv2.destroyWindow('dumm2')


