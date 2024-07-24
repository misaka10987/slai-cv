


import cv2
import numpy as np

# THRESH_TOZERO: turns anything below the threshold to zero, leaves above values alone
# THRESH_BINARY: anything below threshold becomes zero, anything above becomes maxval (an input)
# THRESH_BINARY_INV: flip the previous: below threshold goes to maxval, above it goes to zero
# THRESH_TRUNC: anything above the threshold is set to the threshold, leaves below values alone


beach = cv2.imread("SampleImages/beachBahamas.jpg")
mountain = cv2.imread("SampleImages/grandTetons.jpg")

# state that images read in fine
assert beach is not None
assert mountain is not None


# hsvImg = cv2.cvtColor(beach, cv2.COLOR_BGR2HSV)

# threshed = cv2.inRange(beach, (180, 20, 20), (220, 235, 235))

myMask = np.zeros(beach.shape, beach.dtype)
cv2.rectangle(myMask, (100, 100), (350, 200), (255, 255, 255), -1)
newBeach = cv2.bitwise_and(beach, myMask)
cv2.imshow("Beach", beach)
# cv2.imshow("threshed", threshed)
cv2.imshow("masked", newBeach)
cv2.waitKey()


# 
# grayBeach = cv2.cvtColor(beach, cv2.COLOR_RGB2GRAY)
# grayMt = cv2.cvtColor(mountain, cv2.COLOR_RGB2GRAY)
# 
# for t in range(0, 255, 25):
#     res, thresh = cv2.threshold(grayMt, t, 255, cv2.THRESH_BINARY)
#     newMt = cv2.bitwise_and(mountain, mountain, mask=thresh)
#     cv2.putText(thresh,  str(t), (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, 255)
# 
#     cv2.imshow("Thresholded", thresh)
#     cv2.imshow("New Mountain", newMt)
#     cv2.waitKey()


