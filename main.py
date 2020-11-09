from pypylon import pylon
import cv2 as cv
import distDetect

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

converter = pylon.ImageFormatConverter()
# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

while cam_L.IsGrabbing() and cam_R.IsGrabbing():
    grabResultL = cam_L.RetrieveResult(
        5000, pylon.TimeoutHandling_ThrowException)
    grabResultR = cam_R.RetrieveResult(
        5000, pylon.TimeoutHandling_ThrowException)

    if grabResultL.GrabSucceeded() and grabResultR.GrabSucceeded():
        # Access the image data
        image0 = converter.Convert(grabResultL)
        img0 = image0.GetArray()

        image1 = converter.Convert(grabResultR)
        img1 = image1.GetArray()

        img0, img1 = distDetect.rectifyImage(img0, img1)

        vimg0 = cv.resize(img0, None, fx=0.25, fy=0.25)
        vimg1 = cv.resize(img1, None, fx=0.25, fy=0.25)
        cv.imshow('cam_L', vimg0)
        cv.imshow('cam_R', vimg1)

        k = cv.waitKey(1)
        if k == 27:  # ESC
            break
    grabResultL.Release()
    grabResultR.Release()

# Releasing the resource
cam_L.StopGrabbing()
cam_R.StopGrabbing()

cv.destroyAllWindows()
