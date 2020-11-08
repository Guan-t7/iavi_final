import numpy as np
import cv2 as cv
import glob


def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1, 2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]], -1, (0, 255, 0), -3)
    # draw pillars in blue color
    for i, j in zip(range(4), range(4, 8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]), (255), 3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]], -1, (0, 0, 255), 3)
    return img

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
hp, vp = (6, 7)
objp = np.zeros((hp*vp, 3), np.float32)
objp[:, :2] = np.mgrid[0:hp, 0:vp].T.reshape(-1, 2)
# Intrinsic parameters: camera matrix and distCoeffs
mtx = np.loadtxt(r'mtx.txt')
dist = np.loadtxt(r'dist.txt').reshape(1,5)

images = glob.glob('images3/*.bmp')
for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Arrays to store object points and image points
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.
    # Find the chess board corners in img
    ret, corners = cv.findChessboardCorners(gray, (hp, vp), cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
                                            + cv.CALIB_CB_FAST_CHECK)
    # If found,
    if ret == True:
        # add object points, image points (after refining them)
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners)
        # solve for the extrinsic ones
        retval, rvec, tvec = cv.solvePnP(objpoints[-1], imgpoints[-1], mtx, dist)
        # project 3D points to image plane
        axis = np.float32([[0, 0, 0], [0, 3, 0], [3, 3, 0], [3, 0, 0],
                           [0, 0, -3], [0, 3, -3], [3, 3, -3], [3, 0, -3]])
        imgpts, _ = cv.projectPoints(axis, rvec, tvec, mtx, dist)
        img = draw(img, corners2, imgpts)

        img = cv.resize(img, (1024, 800))
        cv.imshow("test", img)
        cv.waitKey(500)
        # break
    else:
        print(fname, "failed")
cv.destroyAllWindows()

