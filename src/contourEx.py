
import cv2
import numpy as np

img = cv2.imread("SampleImages/wildColumbine.jpg")
(wid, hgt, dep) = img.shape

thresh = cv2.inRange(img, (0, 0, 150), (255, 255, 255))
cv2.imshow("thresh", thresh)
contours, hierarch = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)


for cont in contours:
    area = cv2.contourArea(cont)
    if (area > 500) and (area < 0.9 * wid * hgt):
        copy = img.copy()
        cv2.drawContours(copy, [cont], -1, (255, 0, 0), 2)

        br = cv2.boundingRect(cont)
        cv2.rectangle(copy, (br[0], br[1]), (br[0] + br[2], br[1] + br[3]), (0, 0, 255), 2)
        ell = cv2.fitEllipse(cont)
        cv2.ellipse(copy, ell, (0, 255, 255), 2)

        ch = cv2.convexHull(cont)
        cv2.drawContours(copy, [ch], -1, (255, 255, 0), -1)

        (center, radius) = cv2.minEnclosingCircle(cont)
        print(center, radius)
        center = (int(center[0]), int(center[1]))
        radius = int(radius)
        cv2.circle(copy, center, radius, (0, 255, 0), 2)

        cv2.imshow("Image", copy)
        cv2.waitKey()



