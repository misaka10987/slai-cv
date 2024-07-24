import random
import cv2
import numpy as np

img = cv2.imread("SampleImages/wildColumbine.jpg")
displayCopy = img.copy()
(hgt, wid, _) = img.shape
colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
          (255, 255, 0), (255, 0, 255), (0, 255, 255),
          (128, 0, 0), (0, 128, 0), (0, 0, 128)]

# First, this finds the outline of the flower as a contour

grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
res, threshed = cv2.threshold(grayImg, 95, 255, cv2.THRESH_BINARY)

im, conts, hier = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(displayCopy, conts, -1, random.choice(colors), 2)
cv2.imshow("Contours", displayCopy)
cv2.waitKey(0)

# Now, we get to the fun part. We want to replace everything except the
# flower with a new, blue background. This could be

# 1. Prepare the background picture:

# If you wanted a uniform color:
bg1 = np.zeros(img.shape, img.dtype)
bg1[:,:,0] = 242 # set blue channel to 242
bg1[:,:,1] = 182 # set green channel to 182
bg1[:,:,2] = 145 # set red channel to 145

# If you wanted a different picture as the background
otherPic = cv2.imread('SampleImages/beachBahamas.jpg')
(otherH, otherW, _) = otherPic.shape
cropX = otherW - wid - 1
cropY = otherH - hgt - 1
bg2 = otherPic[cropY:cropY+hgt, cropX:cropX+wid,:]

bg = bg2

# 2. Make a mask, a black-and-white image, that is black where the flower
#    is and white everywhere else, and make its complement
mask1 = np.zeros((hgt, wid), img.dtype)
cv2.drawContours(mask1, conts, -1, 255, -1)
mask2 = 255 - mask1

cv2.imshow("Mask1", mask1)
cv2.imshow("Mask2", mask2)

cv2.waitKey()

# 3. Use mask1 to create a picture that contains just the flower and is
#    black everywhere else

part1 = cv2.bitwise_and(img, img, mask=mask1)

# 4. Use mask2 to create a picture that is the background everywhere
#    except where the flower was, and it's black where the flower was

part2 = cv2.bitwise_and(bg, bg, mask=mask2)

cv2.imshow("Part1", part1)
cv2.imshow("Part2", part2 )
cv2.waitKey()

# 5. Make the final image by adding the two parts

final = part1 + part2

cv2.imshow("FINAL", final)

cv2.waitKey()

cv2.destroyAllWindows()
