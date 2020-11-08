from pypylon import pylon
import numpy as np

camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.Open()
camera.PixelFormat.SetValue('YCbCr422_8')
camera.GainAuto.SetValue("Off")
camera.ExposureAuto.SetValue("Off")
camera.BalanceWhiteAuto.SetValue("Off")

camera.StartGrabbing(pylon.GrabStrategy_OneByOne)

exposuretimes = np.linspace(10, 100000, 15)
gains = [0.0, 1.02305, 1.9382, 2.766054, 3.521825, 4.217067, 4.860761, 5.460025, 6.0206, 6.547179, 7.04365, 7.513272, 7.9588, 8.382586, 8.786654,
         9.172757, 9.542425, 9.897, 10.237667, 10.565476, 10.881361, 11.18616, 11.480625, 11.765434, 12.0412, 12.308479, 12.567779, 12.819561, 13.06425, 13.302235,
         13.533872, 13.9794, 14.403186, 14.807254, 15.193357, 15.563025, 15.9176, 16.258267, 16.586075, 16.901961, 17.20676, 17.501225, 17.786034, 18.0618,
         18.329079, 18.588379, 18.840161, 19.08485, 19.322835, 19.554472, 19.780092, 20.0, 20.214477, 20.423786, 20.628169, 20.827854, 21.02305, 21.213957,
         21.400757, 21.583625, 21.762722, 21.9382, 22.110204, 22.278867, 22.444318, 22.606675, 22.766054, 22.922561, 23.076297, 23.22736, 23.37584, 23.521825,
         23.665397, 23.806634, 23.945611, 24.014275]

for gain in gains:
    for exposuretime in exposuretimes:
        # Request the gain
        camera.Gain.SetValue(gain)
        gain = camera.Gain.GetValue()
        # Request the Exposure Time
        camera.ExposureTime.SetValue(exposuretime)
        exposuretime = camera.ExposureTime.GetValue()

        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
        if grabResult.GrabSucceeded():
            # Access the image data.
            img = grabResult.Array
            print(f"Mean of Y, Std of Y with exp_t {exposuretime} and gain {gain}: ",
                  np.mean(img[:, :, 0], axis=(0, 1)), np.std(img[:, :, 0], axis=(0, 1)))
        grabResult.Release()

camera.Close()
