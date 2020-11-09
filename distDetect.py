import numpy as np
from pypylon import pylon
import cv2 as cv
import glob
import camera_configs as cc


def rectifyImage(imgL, imgR):
    imgL_rectified = cv.remap(imgL, cc.left_map1,  cc.left_map2,  cv.INTER_LINEAR)
    imgR_rectified = cv.remap(imgR, cc.right_map1, cc.right_map2, cv.INTER_LINEAR)
    return imgL_rectified, imgR_rectified
    
def getXYZ(img1, img2, p):
    imgL = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    imgR = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    num = 4
    blockSize = 11
    stereo = cv.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
    disparity = stereo.compute(imgL, imgR)
    # disparity.convertTo(dispf, cv.CV_32F, 1.0/16.0)
    dispf = np.float32(disparity)
    dispf = dispf * 1.0/16.0
    
    x = p[0]
    y = p[1]
    z = dispf[p[1],p[0]]
    xyzw = np.array([[x],[y],[z],[1.0]])
    XYZW = np.matmul(cc.Q, xyzw)
    XYZ = XYZW[:3]/XYZW[3][0]
    
    return XYZ

def getDistance(imgL, imgR, p1, p2):
    XYZ1 = getXYZ(imgL, imgR, p1)
    XYZ2 = getXYZ(imgL, imgR, p2)
    return np.sqrt(np.sum((XYZ1-XYZ2)**2))

    
