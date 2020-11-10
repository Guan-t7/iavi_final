# camera_configs.py
import cv2 as cv
import numpy as np


left_camera_matrix = np.array([[3.749850506498403e+03, -7.954438948815016, 1.351094259002049e+03],
                                [0.,3.743256725596497e+03, 1.069588318627937e+03],
                                [0., 0., 1.]])
left_distortion = np.array([[-0.5412, -0.979, -0.0072, -0.0031, 9.3304]])


right_camera_matrix = np.array([[3.773311983068539e+03, 5.93749826900728, 1.300765190527206e+03],
                               [0., 3.767402065791780e+03, 1.075758906915741e+03],
                               [0., 0., 1.]])
right_distortion = np.array([[-0.5122, -0.8986, -0.0036, -0.0013, 2.0603]])

# om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量
# R = cv.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
R = np.array([0.9996,0.015,-0.0231],[-0.0149,0.9999,0.0056],[0.0232,-0.0052,0.9997])
T = np.array([-144.5303,1.7837,0.6393]) # 平移关系向量

size = (2592, 1944) # 图像尺寸

# 进行立体校正
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv.stereoRectify(left_camera_matrix, left_distortion,
                                                                 right_camera_matrix, right_distortion, 
                                                                 size, R, T)
# 计算校正map
left_map1, left_map2   = cv.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, 
                                                    size, cv.CV_16SC2)
right_map1, right_map2 = cv.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, 
                                                    size, cv.CV_16SC2)