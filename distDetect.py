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
    # stereo = cv.StereoBM_create(numDisparities=16*num, blockSize=blockSize)
    '''
    window_size = 18
    min_disp = 48
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )
    '''
    
    opencv_measure_version = int(cv.__version__.split('.')[0])
    windowSize = 7
    minDisp = 10
    numDisp = 250 - minDisp
    
    # for OpenCV3
    stereo = cv.StereoSGBM_create(
            minDisparity=minDisp,
            numDisparities=numDisp,
            blockSize=16,
            P1=8*3*windowSize**2,
            P2=32*3*windowSize**2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )


    # calculate histogram
    imtGrayL = cv.equalizeHist(imgL)
    imtGrayR = cv.equalizeHist(imgR)
    
    # through gausiann filter
    imgGrayL = cv.GaussianBlur(imtGrayL, (5, 5), 0)
    imgGrayR = cv.GaussianBlur(imtGrayR, (5, 5), 0)
    
    # calculate disparity
    disparity = stereo.compute(imgGrayL, imgGrayR).astype(np.float32)/16
    disparity = (disparity - minDisp) / numDisp
    disparity1 = cv.resize(disparity, None, fx=0.25, fy=0.25)
    cv.imshow("disparity", disparity1)
    
    '''
    disparity = stereo.compute(imgL, imgR)
    disparity = cv.convertScaleAbs(disparity, alpha=255/disparity.max())
    disp_colored = cv.applyColorMap(disparity, cv.COLORMAP_JET)
    disp_colored = cv.resize(disp_colored, None, fx=0.25, fy=0.25)
    cv.imshow('disparity', disp_colored)
    
    '''
    # disparity.convertTo(dispf, cv.CV_32F, 1.0/16.0)
    # dispf = np.float32(disparity)
    # dispf = dispf * 1.0/16.0

    x = p[0]
    y = p[1]
    z = disparity[p[1],p[0]]
    xyzw = np.array([[x],[y],[z],[1.0]])
    XYZW = np.matmul(cc.Q, xyzw)
    XYZ = XYZW[:3]/XYZW[3][0]
    
    return XYZ
    
    #return np.array([[1],[2],[3]])

def getDistance(imgL, imgR, p1, p2):
    XYZ1 = getXYZ(imgL, imgR, p1)
    XYZ2 = getXYZ(imgL, imgR, p2)
    return np.sqrt(np.sum((XYZ1-XYZ2)**2))

    
