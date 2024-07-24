import cv2

img = cv2.imread("./res/img/wildColumbine.jpg")

# Filter types: MORPH_DILATE, MORPH_ERODE, MORPH_OPEN, MORPH_CLOSE, MORPH_TOPHAT, MORPH_BLACKHAT, MORPH_GRADIENT
morphType = cv2.MORPH_DILATE

# Filter shapes: MORPH_RECT, MORPH_ELLIPSE, MORPH_CROSS, plus you can define your own
morphShape = cv2.MORPH_ELLIPSE

kernelObj = cv2.getStructuringElement(morphShape, (9, 9))
newImg = cv2.morphologyEx(img, morphType, kernelObj)

cv2.imshow("Original", img)
cv2.imshow("Morphed", newImg)
cv2.waitKey()
