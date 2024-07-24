
import cv2
import numpy as np



fast = cv2.FastFeatureDetector_create()


img = cv2.imread("SampleImages/PuzzlesAndGames/puzzle2.jpg")
kp1 = fast.detect(img, None)
# print(kp1)
img2 = cv2.drawKeypoints(img, kp1, None, (0, 255, 0))

print("Threshold:", fast.getThreshold())
print("nonmaxSuppression:", fast.getNonmaxSuppression())
print("neighborhood:", fast.getType())
print("Total Keypoints with nonmaxSuppression:", len(kp1))

fast.setNonmaxSuppression(False)

kp2 = fast.detect(img, None)
img3 = cv2.drawKeypoints(img, kp2, None, (0, 255, 0))
print("Total Keypoints without nonmaxSuppression:", len(kp2))

cv2.imshow("Keypoints nonmax", img2)
cv2.imshow("Keypoints, no supp", img3)
cv2.waitKey()

cv2.destroyAllWindows()



cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    res, frame = cap.read()
    if res:
        cv2.imshow("Original", frame)
        kp = fast.detect(frame, None)
        img2 = cv2.drawKeypoints(frame, kp, None, (0, 0, 255))
        cv2.imshow("KeyFrame", img2)
        v = cv2.waitKey(20)
        c = chr(v&0xFF)
        if c == 'q':
            break


