import cv2

img1 = cv2.imread('./data/night_chess.png',0)
# img1 = cv2.resize(img1, (400, 400))
# img1 = cv2.undistort(img1)


def chessboard(image, pattern_size=(6,7)):
    status, corners = cv2.findChessboardCorners(image, pattern_size, flags=\
    cv2.CALIB_CB_NORMALIZE_IMAGE |
    cv2.CALIB_CB_ADAPTIVE_THRESH |
    cv2.CALIB_CB_FILTER_QUADS
                                                )
    print ("STATUS: [{}]".format(status))
    if status:
        mean = corners.sum(0)/corners.shape[0]
        # mean is [[x,y]]
        return mean[0], corners
    else:
        return None

a, b = chessboard(img1)

founded_board = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
for corner in b:
    founded_board = cv2.circle(
                  img = founded_board,
        center=(corner[0][0], corner[0][1]),
        radius=5,
                  color=(0, 255, 50 ),
                  thickness=2)
    # print ( corner.shape )



while True:
    cv2.imshow("chess", founded_board)
    ch = cv2.waitKey(1)
    if ch == 27:  # ESC to out
        break

cv2.destroyAllWindows()


# img1 = cv2.rectangle(
#                   img = img1,
#                   pt1 = eyebox_corner1,
#                   pt2 = eyebox_corner2,
#                   color=(0, 255, 0),
#                   thickness=2)

# print (b)