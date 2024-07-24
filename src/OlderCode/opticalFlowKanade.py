__author__ = 'sfox2'


import numpy as np
import cv2

cap = cv2.VideoCapture(0)


# Create some random colors
color = np.random.randint(0, 255, (100, 3))


# Wait until user is ready to start tracking, user hits space to go on
while True:
    ret, old_frame = cap.read()

    cv2.imshow("Frame", old_frame)
    c = chr(cv2.waitKey(10) & 0xFF)
    if c == ' ':
        break

# Grap features on most recent frame using Shi-Tomasi
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray,
                             mask=None,
                             maxCorners=300,
                             qualityLevel=0.3,
                             minDistance=7,
                             blockSize=7)

# Create a mask image for drawing purposes
mask = np.zeros(old_frame.shape, old_frame.dtype)

# Loop over video while tracking
while True:
    ret, frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None,
                                           winSize=(15, 15),
                                           maxLevel=2,
                                           criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    # draw the tracks
    for i , (new, old) in enumerate(zip(good_new, good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)
    img = cv2.add(frame,mask)

    cv2.imshow('Frame', img)
    k = chr(cv2.waitKey(30) & 0xff)
    if k == 'q':
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()