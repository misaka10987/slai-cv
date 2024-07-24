

import cv2

topCell = None


def callback(data):
    global topCell
    
    strng = data.data
    topCell = .... 

vidCap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

ret, frame = vidCap.read()
prevFrame = frame

while True:
    ret, nextFrame = vidCap.read()

    diffPic = cv2.absdiff(prevFrame, nextFrame)
    prevFrame = nextFrame

    cv2.imshow("Difference", diffPic)
    x = cv2.waitKey(20)

    ch = chr(x & 0xFF)
    if ch == 'q':
        break

cv2.destroyAllWindows()
vidCap.release()


