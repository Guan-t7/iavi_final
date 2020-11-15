import numpy as np
import cv2 as cv
import glob
import camera_configs as cc


def rectifyImage(imgL, imgR):
    imgL_rectified = cv.remap(imgL, cc.left_map1,  cc.left_map2,  cv.INTER_LINEAR)
    imgR_rectified = cv.remap(imgR, cc.right_map1, cc.right_map2, cv.INTER_LINEAR)
    return imgL_rectified, imgR_rectified
    
def getXYZ(disparity, p):
    
    x = p[0]
    y = p[1]
    z = disparity[p[1],p[0]]
    xyzw = np.array([[x],[y],[z],[1.0]])
    XYZW = np.matmul(cc.Q, xyzw)
    XYZ = XYZW[:3]/XYZW[3][0]
    
    return XYZ
    
def getDistance(disparity, p1, p2):
    
    XYZ1 = getXYZ(disparity, p1)
    XYZ2 = getXYZ(disparity, p2)

    return np.sqrt(np.sum((XYZ1-XYZ2)**2))

    
