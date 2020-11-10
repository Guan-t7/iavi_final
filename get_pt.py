import cv2 as cv
import distDetect

phase = 'idle'
imgpts = []

# mouse callback function
def draw(event, x, y, flags, param):
    global phase, imgpts
    if event == cv.EVENT_LBUTTONDOWN:
        imgpts.append((x, y))
        phase = f'point {len(imgpts)} acquired'


img0 = cv.imread("..\image_\calib1\R15.jpg")
cv.namedWindow('cam')
cv.setMouseCallback('cam', draw)

while True:
    for pt in imgpts:
        img0 = cv.circle(img0, pt, 5, (0, 0, 255),
                        thickness=1, lineType=cv.LINE_AA)
    cv.setWindowTitle('cam', phase)
    
    cv.imshow('cam', img0)
    k = cv.waitKey(1)
    if k == 27:  # ESC
        break
