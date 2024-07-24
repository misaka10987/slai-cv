


import cv2

videoCap = cv2.VideoCapture(0)
print("Video ready")

while True:
    gotIt, frame = videoCap.read()
    if not gotIt:
        break
    cv2.imshow("Video Feed", frame)
    x = cv2.waitKey(30)
    ch = chr(x&0xFF)
    if ch == 'q':
        break

videoCap.release()

