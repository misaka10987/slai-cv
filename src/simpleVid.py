

import cv2
import numpy as np
import random


vidCap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    gotFrame, frame = vidCap.read()
    if not gotFrame:
        print("no frame")
        break
    (h, w, d) = frame.shape
    cv2.imshow("Video", frame)
    x = cv2.waitKey(30)
    ch = chr(x & 0xFF)
    if ch == 'q':
        break

vidCap.release()
