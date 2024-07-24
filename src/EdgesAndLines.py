
import cv2
import numpy as np


origIm = cv2.imread("SampleImages/chicago.jpg")

assert origIm is not None

grayIm = cv2.cvtColor(origIm, cv2.COLOR_BGR2GRAY)

sobel1 = cv2.Sobel(grayIm, cv2.CV_32F, 1, 0)  # 1st derivative in the x direction
sobel2 = cv2.Sobel(grayIm, cv2.CV_32F, 0, 1)  # 1st derivative in the y direction
sobelComb = cv2.addWeighted(sobel1, 0.5, sobel2, 0.5, 0)

sobelX = cv2.convertScaleAbs(sobel1)
sobelY = cv2.convertScaleAbs(sobel2)
sobelBoth = cv2.convertScaleAbs(sobelComb)
cv2.imshow("Sobel X", sobelX)
cv2.imshow("Sobel Y", sobelY)
cv2.imshow("Sobel Comb", sobelBoth)

cv2.waitKey()

canny = cv2.Canny(origIm, 80, 110)

cv2.imshow("Canny Im", canny)

cv2.waitKey()

