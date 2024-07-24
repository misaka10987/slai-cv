

import cv2
import numpy as np


# Some versions of OpenCV need this to fix a bug
# cv2.ocl.setUseOpenCL(False)

targetImg = cv2.imread("SampleImages/Coins/dollarCoin.jpg")
testImg = cv2.imread("SampleImages/Coins/coins1.jpg")

(hgt, wid, dep) = targetImg.shape

# create an ORB object, that can run the ORB algorithm.
orb = cv2.ORB_create()  # some versions use cv2.ORB() instead
bfMatcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

while True:
    for ang in range(0, 360, 90):
        rotMatrix = cv2.getRotationMatrix2D((wid/2, hgt/2), ang, 1)
        target2 = cv2.warpAffine(targetImg, rotMatrix, (wid, hgt))
        kp1, des1 = orb.detectAndCompute(target2, None)
        kp2, des2 = orb.detectAndCompute(testImg, None)

        outImage = np.zeros((1, 1, 1), np.uint8)

        targetKeys = cv2.drawKeypoints(target2, kp1, outImage)
        cv2.imshow("Target Keypoints", targetKeys)

        matches1 = bfMatcher.match(des1, des2)
        matches1.sort(key=lambda x: x.distance)  # sort by distance

        # count matches1 with distance less than threshold
        i = 0
        for i in range(len(matches1)):
            if matches1[i].distance > 70.0:
                break

        # draw matches within given distance
        outImage = np.zeros((1, 1, 1), np.uint8)
        img3 = cv2.drawMatches(target2, kp1, testImg, kp2, matches1[:i], outImage)
        cv2.imshow("Matches1", img3)

        x = cv2.waitKey(0)
        if chr(x&0xFF) == 'q':
            break
    x = cv2.waitKey(0)
    if chr(x&0xFF) == 'q':
        break



cv2.destroyAllWindows()

