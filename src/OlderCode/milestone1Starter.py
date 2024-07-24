
import cv2
import numpy

def mouseResponse(event, x, y, flags, param):
    """This function is a callback that happens when the mouse is used.
    event describes which mouse event triggered the callback, (x, y) is
    the location in the window where the event happened. The other inputs
    may be ignored."""
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(workImg, (x, y), 10, (255, 0, 255), -1)

# read in an image
origImg = cv2.imread("TestImages/SnowLeo1.jpg")
height, width, depth = origImg.shape

# make a copy and set up the window to display it
workImg = origImg.copy()
cv2.namedWindow("Working image")

# assign mouse_response to be the callback function for the Working image window
cv2.setMouseCallback("Working image", mouseResponse)

# Keep re-displaying the window, and look for user to type 'q' to quit
while True:
    cv2.imshow("Working image", workImg)
    x = cv2.waitKey(20)
    ch = chr(x & 0xFF)
    if ch == 'q':
        break

cv2.destroyAllWindows()
