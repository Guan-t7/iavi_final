import cv2 as cv
import numpy as np

drawing = False  # true if mouse is pressed
done = False
ix, iy, fx, fy = -1, -1, -1, -1

# mouse callback function
def draw(event, x, y, flags, param):
    global ix, iy, fx, fy, drawing, done
    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv.EVENT_MOUSEMOVE:
        if drawing == True:
            # cv.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)
            pass
    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        # cv.rectangle(img, (ix, iy), (x, y), (0, 0, 0), -1)
        fx, fy = x, y
        done = True


cv.namedWindow('image')
cv.setMouseCallback('image', draw)

bgr_idx = {'b': 6, 'g': 9, 'r': 8}
cam_bgr = {}
for c, i in bgr_idx.items():
    img_l = cv.imread(f"image2/L{i}.bmp")
    # img_r = cv.imread(f"image2/R{i}.bmp")
    for img in [img_l, ]:  # img_r
        done = False
        cv.setWindowTitle('image', "draw a rect to indicate sampling area")
        img = cv.resize(img, None, fx=0.25, fy=0.25)
        while (1):
            cv.imshow('image', img)
            cv.waitKey(1)
            if done: break
        area = img[iy:fy, ix:fx]
        cam_bgr[c] = np.mean(area, axis=(0, 1))

cam_bgr = np.asarray([col for col in cam_bgr.values()]).T
base = np.diagflat([255, 255, 255])
mat = np.linalg.lstsq(np.linalg.inv(base), np.linalg.inv(cam_bgr))[0]

np.savetxt('mat.txt', mat)