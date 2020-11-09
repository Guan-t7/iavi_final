from pypylon import pylon
import cv2 as cv
import distDetect
import time

phase = 'idle'
imgpts = []
pt_r = 20
fx = fy = 0.25

# mouse callback function
def draw(event, x, y, flags, param):
    global phase, imgpts
    if event == cv.EVENT_LBUTTONDOWN:
        imgpts.append((int(x/fx), int(y/fy))) # original image
        phase = f'point {len(imgpts)} acquired'

            

def init_stereo_cam():
    tup = pylon.TlFactory.GetInstance().EnumerateDevices()

    if tup[0].GetSerialNumber() == "40031419":
        dev_L, dev_R = tup
    else:
        dev_R, dev_L = tup

    cam_L = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(dev_L))
    cam_R = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(dev_R))

    cam_L.Open()
    cam_R.Open()

    # Grabing Continusely (video) with minimal delay
    cam_L.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    cam_R.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    for camera in [cam_L, cam_R]:
        camera.GainAuto.SetValue("Off")
        camera.ExposureAuto.SetValue("Off")
        camera.BalanceWhiteAuto.SetValue("Off")
    return cam_L, cam_R

cam_L, cam_R = init_stereo_cam()

converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

cv.namedWindow('cam_L')
cv.namedWindow('cam_R')
cv.setMouseCallback('cam_L', draw)

while cam_L.IsGrabbing() and cam_R.IsGrabbing():
    grabResultL = cam_L.RetrieveResult(
        500, pylon.TimeoutHandling_ThrowException)
    grabResultR = cam_R.RetrieveResult(
        500, pylon.TimeoutHandling_ThrowException)
    
    print(grabResultL.GrabSucceeded(), grabResultR.GrabSucceeded())

    if grabResultL.GrabSucceeded() and grabResultR.GrabSucceeded():
        # Access the image data
        image0 = converter.Convert(grabResultL)
        img0 = image0.GetArray()

        image1 = converter.Convert(grabResultR)
        img1 = image1.GetArray()

        for pt in imgpts:
            img0 = cv.circle(img0, pt, pt_r, (0, 0, 255),
                             thickness=int(0.2 * pt_r), lineType=cv.LINE_AA)
        cv.setWindowTitle('cam_L', phase)
        cv.setWindowTitle('cam_R', phase)
        # img0, img1 = distDetect.rectifyImage(img0, img1)

        vimg0 = cv.resize(img0, None, fx=fx, fy=fy)
        vimg1 = cv.resize(img1, None, fx=fx, fy=fy)
        cv.imshow('cam_L', vimg0)
        cv.imshow('cam_R', vimg1)

        k = cv.waitKey(1)
        if k == 27:  # ESC
            break
    grabResultL.Release()
    grabResultR.Release()
    time.sleep(0.1)

# Releasing the resource
cam_L.StopGrabbing()
cam_R.StopGrabbing()

cv.destroyAllWindows()


