import numpy as np
import cv2 as cv
import time


#TODO projector brightness to maximum
def scan_window(FPS=10, line_width=31):
    # Create a black image, a window
    img = np.zeros((1080, 1920, 3), np.uint8)
    win_name = "scan_window"
    cv.namedWindow(win_name, cv.WINDOW_FULLSCREEN)
    # Create scan pattern
    scan_pattern = np.zeros((1080, line_width, 3), np.uint8)
    for i in range(line_width):
        scan_pattern[:, i] = int(i / line_width * 255)
    scan_pattern = cv.applyColorMap(scan_pattern, cv.COLORMAP_RAINBOW)
    # init scan
    right = line_width
    while True:
        cv.imshow(win_name, img)
        k = cv.waitKey(0)
        if k == 27:  # ESC
            return
        if k == ord('f'):
            break
    # start scan
    while right <= 1920:
        # scan line to project
        proj_img = img.copy()
        proj_img[:, right - line_width:right] = scan_pattern
        cv.imshow(win_name, proj_img)
        k = cv.waitKey(int(1000 / FPS))
        if k == 27:  # ESC
            break
        if k == ord('r'):
            right = 0
        right += line_width



if __name__ == '__main__':
    print(__doc__)
    scan_window()
    cv.destroyAllWindows()
