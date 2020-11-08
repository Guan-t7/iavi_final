import cv2 as cv
import numpy as np

mat = np.loadtxt('mat.txt')
img_l = cv.imread(f"image2/L9.bmp")
x,y,_ = img_l.shape
img_l = img_l.reshape(-1,3)
img_l = mat @ img_l.T
img_l = img_l.T.reshape(x, y, 3)
img_l[img_l < 0] = 0
img_l[img_l > 255] = 255
img_l = img_l.astype('uint8')
img_l = cv.resize(img_l, None, fx=0.5, fy=0.5)
cv.imshow('result', img_l)
cv.waitKey(0)
