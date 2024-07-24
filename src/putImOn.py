"""
This program displays the video feed on one side of a bigger window, and lets the
user select from three pictures to display on the left size.
"""
import cv2
import numpy as np
import random

mountains = cv2.imread("SampleImages/grandTetons.jpg")
city = cv2.imread("SampleImages/chicago.jpg")
flower = cv2.imread("SampleImages/wildColumbine.jpg")

mtSized = cv2.resize(mountains, (300, 300))
citySized = cv2.resize(city, (300, 300))
flowSized = cv2.resize(flower, (300, 300))


bigScreen = np.zeros((350, 675, 3), np.uint8)
bigScreen[:,:,:] = (160, 0, 100)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frameSized = cv2.resize(frame, (300, 300))
    bigScreen[25:325, 350:650, :] = frameSized
    cv2.imshow("Game", bigScreen)
    res = cv2.waitKey(30)
    ch = chr(res & 0xFF)
    if ch == 'q':
        break
    elif ch == 'm':
        bigScreen[25:325, 25:325, :] = mtSized
    elif ch == 'c':
        bigScreen[25:325, 25:325, :] = citySized
    elif ch == 'f':
        bigScreen[25:325, 25:325, :] = flowSized


cap.release()
