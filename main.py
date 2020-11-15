from pypylon import pylon
import cv2 as cv
import distDetect
import concurrent.futures
from functools import partial
import numpy as np

phase_dict = {0: 'LBUTTON click to give point 1',
              1: 'give second point', 2: 'calculating distance: '}
imgpts = []
trackers = []
# GUI config
pt_r = 20
fx = fy = 0.25
cv.circle = partial(cv.circle, color=(0, 0, 255),
                    thickness=int(0.2 * pt_r), lineType=cv.LINE_AA)
cv.line = partial(cv.line, color=(0, 0, 255),
                  thickness=int(0.2 * pt_r), lineType=cv.LINE_AA)

# mouse callback function
#! point coordinate based on original image
def draw(event, x, y, flags, param):
    global phase, imgpts
    draw.x, draw.y = int(x/fx), int(y/fy)
    if len(imgpts) != 2:
        if event == cv.EVENT_LBUTTONDOWN:
            imgpts.append((int(x/fx), int(y/fy))) 
            phase = f'point {len(imgpts)} acquired'


draw.x = draw.y = 0



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
        # print(camera.Gain.GetValue())
        # print(camera.ExposureTime.GetValue())
        # print(camera.BalanceRatio.GetValue())
        camera.AcquisitionFrameRate.SetValue(6)
    
    return cam_L, cam_R


def main():
    global phase_dict, imgpts, trackers, pt_r, fx, fy
    # background process executor for calculating distance
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
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
        
        # print(grabResultL.GrabSucceeded(), grabResultR.GrabSucceeded())

        if grabResultL.GrabSucceeded() and grabResultR.GrabSucceeded(): # update image
            image0 = converter.Convert(grabResultL)
            ori_img0 = image0.GetArray()

            image1 = converter.Convert(grabResultR)
            ori_img1 = image1.GetArray()

        img0, img1 = ori_img0.copy(), ori_img1.copy()  # start proc
        img0, img1 = distDetect.rectifyImage(img0, img1)

        bbox_r = 2 * pt_r
        for i in range(len(imgpts)):
            try:
                success, bbox = trackers[i].update(img0)
            except IndexError:
                trackers.append(cv.TrackerCSRT_create())
                bbox = imgpts[i][0] - bbox_r, imgpts[i][1] - \
                    bbox_r, 2*bbox_r, 2*bbox_r
                trackers[i].init(img0, bbox)
            else:
                if success:
                    imgpts[i] = int(bbox[0] + bbox[2]/2), int(bbox[1] + bbox[3]/2)


        mousept = (draw.x, draw.y)
        if len(imgpts) == 0:
            img0 = cv.circle(img0, mousept, pt_r)
        elif len(imgpts) == 1:
            img0 = cv.circle(img0, imgpts[0], pt_r)
            img0 = cv.circle(img0, mousept, pt_r)
            img0 = cv.line(img0, imgpts[0], mousept)
        elif len(imgpts) == 2:
            img0 = cv.circle(img0, imgpts[0], pt_r)
            img0 = cv.circle(img0, imgpts[1], pt_r)
            img0 = cv.line(img0, imgpts[0], imgpts[1])

        if len(imgpts) == 2:
            if phase_dict[2] == 'calculating distance: ':
                try:
                    if f.done():
                        phase_dict[2] += f'{f.result()}'
                        del f
                except NameError:
                    f = executor.submit(distDetect.getDistance, img0, img1, imgpts[0], imgpts[1])
            
        
        cv.setWindowTitle('cam_L', phase_dict[len(imgpts)])
        # cv.setWindowTitle 'cam_R'

        vimg0 = cv.resize(img0, None, fx=fx, fy=fy)
        vimg1 = cv.resize(img1, None, fx=fx, fy=fy)
        cv.imshow('cam_L', vimg0)
        cv.imshow('cam_R', vimg1)

        k = cv.waitKey(1)
        if k == 27:  # ESC
            break
        if k == ord('r'):
            imgpts = []
            trackers = []
            phase_dict[2] = 'calculating distance: '
        grabResultL.Release()
        grabResultR.Release()

    # Releasing the resource
    cam_L.StopGrabbing()
    cam_R.StopGrabbing()

    cv.destroyAllWindows()

if __name__ == '__main__':
    main()
