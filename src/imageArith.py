


import cv2
import numpy as np

im1 = cv2.imread("SampleImages/canyonlands.jpg")
# im2 = cv2.imread("SampleImages/mushrooms.jpg")

cv2.imshow("Canyonlands", im1)
# cv2.imshow("Mushrooms", im2)


cv2.waitKey()

justFifty = np.zeros(im1.shape, im1.dtype) + 50

im3 = im1 + 50        # Using Numpy's addition
im4 = cv2.add(im1, justFifty)

print("original")
print(im1[0:3, 0:3,:])
print("added")
print(im4[0:3, 0:3, :])

cv2.imshow("Add with Numpy", im3)
cv2.imshow("Add with CV2", im4)

cv2.waitKey()






added = cv2.add(im1, im2)
blend = cv2.addWeighted(im1, 0.8, im2, 0.2, 0)
cv2.imshow("Added", added)
cv2.imshow("Blended", blend)
cv2.waitKey()




add50 = np.zeros(im1.shape, im1.dtype) + 50
im4 = cv2.add(im1, add50)
