


import cv2
import numpy as np

def doCoinFind(image):
    ret, colThresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imshow("Color thresh", colThresh)

    grayIm = cv2.cvtColor(colThresh, cv2.COLOR_BGR2GRAY)
    ret, grayThresh = cv2.threshold(grayIm, 200, 255, cv2.THRESH_BINARY)
    ret, thresh2 = cv2.threshold(grayIm, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("grayThresh1", grayThresh)
    cv2.imshow("grayThresh2", thresh2)

    comb = cv2.bitwise_or(grayThresh, thresh2)
    cv2.imshow("COMBINED", comb)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opeGray = cv2.morphologyEx(comb, cv2.MORPH_DILATE, kernel,iterations=2)
    cv2.imshow("DILATED", opeGray)

    res, cont, hier = cv2.findContours(opeGray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    finalIm = cv2.drawContours(image, cont, -1, (0, 255, 0), 2)
    return finalIm, cont

def runImage(filename):
    im1 = cv2.imread(filename)
    cv2.imshow("Orig1", im1)
    cv2.waitKey()
    foundIm1, cont1 = doCoinFind(im1)
    cv2.imshow("Found1", foundIm1)
    cv2.waitKey()


runImage("SampleImages/Coins/coins1.jpg")
runImage("SampleImages/Coins/coins2.jpg")
runImage("SampleImages/Coins/coins4.jpg")
runImage("SampleImages/Coins/coins6.jpg")
cv2.destroyAllWindows()



# coinImg = cv2.imread("TestImages/coins1.jpg")
#
# cv2.imshow("Original", coinImg)
#
# (ht, wd, dp) = coinImg.shape
#
# foobar = np.logical_or(coinImg[:,:,1] > coinImg[:,:,2], coinImg[:,:,1] > coinImg[:,:,0])
# print(foobar)
#
# fooImg = np.uint8(foobar) * 255
# cv2.imshow("Foobar", fooImg)
#
# blank = np.zeros((ht, wd, 1), np.uint8)
# blank2 = 255 * np.ones((ht, wd, 1), np.uint8)
#
# mask = cv2.merge((blank, blank2, blank))
#
# newIm = cv2.absdiff(coinImg, mask)
#
# cv2.imshow("New", newIm)
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()


