import cv2



#======== part for output
sample = "./baboon.jpg"
projected_picture_name = 'projector_test'
output = cv2.imread(sample)


#part for input

while True:
    cv2.imshow(projected_picture_name, output)
    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break

cv2.destroyAllWindows()