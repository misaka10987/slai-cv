

import cv2
import numpy as np


bkgnd = cv2.imread("SampleImages/landscape1.jpg")
eye = cv2.imread("SampleImages/GoogleyEye.png")
(eHgt, eWid, dep) = eye.shape

# Make two masks using threshold, one for the eye, one for its background
gEye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gEye, 200, 255, cv2.THRESH_BINARY)
bgMask = thresh
fgMask = 255 - thresh
cv2.imshow("bgMask", bgMask)
cv2.imshow("fgMask", fgMask)

bg = bkgnd[400:400+eHgt, 400:400+eWid,:]
bg = cv2.bitwise_and(bg, bg, mask=bgMask)
fg = cv2.bitwise_and(eye, eye, mask=fgMask)

# At this point, bg has black where the eye will go, and the background everywhere else
# At this point, fg has the eye, with black where the background will be
cv2.imshow("background part where eye is", bg)
cv2.imshow("foreground part where eye is", fg)

bkgnd[400:400+eHgt, 400:400+eWid,:] = bg + fg

cv2.imshow("Final", bkgnd)

cv2.waitKey(0)