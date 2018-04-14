import cv2 as cv
import numpy as np


def get_camera_box_angles():
    pass

def get_picture_angles_normal_vector_in_3d():
    pass

def get_center_of_an_image(img):
    x,y,depth = img.shape
    center_x = x // 2
    center_y = y // 2
    return (center_x,center_y)

def twoD_to_threeD(matrix):
    pass


def warp_the_image_in_3d(img, M_from, M_to):

    rows, cols, ch = img.shape

    M_from_3 = []
    M_to_3 = []
    # print(M_from.shape[0])
    # M_to_3 = np.array(M_to.size)
    for el in range(M_from.shape[0]):
        M_from_3.append( np.array( [M_from[el][0],  M_from[el][1], 0] ) )
        M_to_3.append( np.array([M_to[el][0], M_to[el][1], 0]) )
    # false_vector=  (M_from_3[0] + M_from_3[1]) // 2
    # M_from_3.append(false_vector)

    M_from_3 = np.float64(M_from_3)
    # false_vector=  (M_to_3[0] + M_to_3[1]) // 2
    # M_to_3.append(false_vector)
    M_to_3 = np.float64(M_to_3)


    retval, out, inliers = cv.cv2.estimateAffine3D(M_from_3, M_to_3)
    print("Retval")
    print(retval)
    print("inliers")
    print(inliers)
    print ("Affine transform matrix is:")
    np.set_printoptions(suppress=True)

    print(out)

    print(M_from)
    print(M_from_3)
    print(M_to)
    print(M_to_3)

    print ("EXPERIMENT!==============")
    res =  np.matmul( out , np.array([50, 200, 0, 1])  )
    print (M_from_3)
    print (res)
    print (M_to_3)
    return out

    # dst = cv.cv2.warpAffine(img, out, (cols, rows))
    # return dst


def warp_the_image(img, M_from, M_to):
    rows, cols, ch = img.shape

    M = cv.cv2.getAffineTransform(M_from, M_to)
    print ("Affine transform matrix is:")
    np.set_printoptions(suppress=True)
    print(M)

    dst = cv.cv2.warpAffine(img, M, (cols, rows))
    return dst


active_picture_name = "Active_pic"
sample_picture_name = "sample_pic"


#
# cv.namedWindow(winName)
# cv.namedWindow(faceName)
cv.namedWindow(active_picture_name)
cv.namedWindow(sample_picture_name)


sample = "./baboon.jpg"
img = cv.imread(sample)

img_rows,img_cols,ch = img.shape # will be global wariables between transitions

center_of_coordinates = get_center_of_an_image(img)
#
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# pts1 = np.float32([[50, 50], [200, 50]])
# pts2 = np.float32([[10, 100], [200, 50]])

print(len(pts1))
for point_index in range (len(pts1)):
    print (pts1[point_index])
    img = \
        cv.circle(
            img=img,
            center=(pts1[point_index][0], pts1[point_index][1]),
            radius=3,
            color=(0, 0, 255),
            thickness=2)

img = cv.circle(
            img=img,
            center=center_of_coordinates,
            radius=3,
            color=(0, 255, 255),
            thickness=2)

pts1_refarding_center = pts1 - [center_of_coordinates[0], center_of_coordinates[1]]
pts2_refarding_center = pts2 - [center_of_coordinates[0], center_of_coordinates[1]]

center_np = np.float32([center_of_coordinates[0], center_of_coordinates[1]])

# Here we need to find such Affine transformation that preserves the center of coordinates

pts1_c = np.append(pts1, np.array([center_np]), axis=0)
pts2_c = np.append(pts2, np.array([center_np]), axis=0)


print(pts1_c)
print(pts2_c)

print(pts1)
print(pts1_refarding_center)

#np.concatenate(pts1, center_np, axis=1)
# pts1, np.array([center_of_coordinates[0], center_of_coordinates[1]])
# pts1_c  = np.concatenate()
print(center_np)


# dst = warp_the_image (img, pts1_c, pts2_c)

dst3d = warp_the_image_in_3d (img, pts1_c, pts2_c)


dst3d_2d = np.array( [
    [dst3d[0][0], dst3d[0][1], dst3d[0][3]],
    [dst3d[1][0], dst3d[1][1], dst3d[1][3]],
])
print (dst3d.shape)
print (dst3d)
print (dst3d_2d)


rows, cols, ch = img.shape

M = cv.cv2.getAffineTransform(pts1, pts2)
print ("Affine transform matrix is:")
np.set_printoptions(suppress=True)
print(M)

dst = cv.cv2.warpAffine(img, M, (cols, rows))
dst3d = cv.cv2.warpAffine(img, dst3d_2d, (cols, rows))
# dst3d_2d = dst3d[]


while True:
    cv.imshow(active_picture_name, dst)
    cv.imshow(sample_picture_name, dst3d)
    ch = cv.waitKey(1)
    if ch == 27:  # ESC to out
        break
cv.destroyAllWindows()



