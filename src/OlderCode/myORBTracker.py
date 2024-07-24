

import cv2
import numpy as np


# Some versions of OpenCV need this to fix a bug
# cv2.ocl.setUseOpenCL(False)

targetImg = cv2.imread("sampleImageORB.jpg")


(hgt, wid, dep) = targetImg.shape
outImage = np.zeros((1, 1, 1), np.uint8)

# create an ORB object, that can run the ORB algorithm.
orb = cv2.ORB_create()  # some versions use cv2.ORB() instead
bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

kp1, des1 = orb.detectAndCompute(targetImg, None)

vidCap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    res, frame = vidCap.read()

    kp2, des2 = orb.detectAndCompute(frame, None)


    targetKeys = cv2.drawKeypoints(targetImg, kp1, outImage)
    cv2.imshow("Target Keypoints", targetKeys)

    matches1 = bfMatcher.match(des1, des2)
    matches1.sort(key=lambda x: x.distance)  # sort by distance

    # count matches1 with distance less than threshold
    i = 0
    for i in range(len(matches1)):
        if matches1[i].distance > 70.0:
            break

    # draw matches within given distance
    img3 = cv2.drawMatches(targetImg, kp1, frame, kp2, matches1[:i], outImage)
    cv2.imshow("Matches1", img3)

    x = cv2.waitKey(30)
    if chr(x&0xFF) == 'q':
        break



cv2.destroyAllWindows()

