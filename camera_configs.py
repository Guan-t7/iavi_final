# camera_configs.py
import cv2 as cv
import numpy as np


left_camera_matrix = np.array([[3.720190840220664e+03, -0.12248184784898, 1.281961846910185e+03],
                                [0., 3.715252397555289e+03, 9.905046367453955e+02],
                                [0., 0., 1.]])
left_distortion = np.array([[-0.5781, 0.6451, -3.6186e-04, -1.0396e-04, -2.5315]])


right_camera_matrix = np.array([[3.743461997010555e+03, 4.551937441802115, 1.298286260418159e+03],
                               [0., 3.732773095344897e+03, 1.011617064328398e+03],
                               [0., 0., 1.]])
right_distortion = np.array([[-0.5538, -0.4112, 0.0033, -9.9302e-04, 4.5406]])

# om = np.array([0.01911, 0.03125, -0.00960]) # 旋转关系向量
# R = cv.Rodrigues(om)[0]  # 使用Rodrigues变换将om变换为R
R = np.array([[0.9996, 0.0057, -0.0282],[-0.0054, 0.9999, 0.0126],[0.0283, -0.0124, 0.9995]])
T = np.array([-72.1701, 1.2193, 2.3877]) # 平移关系向量

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