

import cv2


vidCap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

while True:
    gotFrame, frame = vidCap.read()
    if not gotFrame:
        print("no frame")
        break
    
    cv2.imshow("Video", frame)
    cv2.waitKey(30)
    
vidCap.release()
