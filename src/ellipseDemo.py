"""
File: ellipseDemo.py

This file illustrates the effect of changing parameters in drawing an ellipse.
The ellipse function in OpenCV takes in the following inputs:
img:        the image on which to draw the ellipse
center:     a point (x, y) tuple where the center of the ellipse will be
axes:       a tuple (major, minor) giving the length of the major (longer) and minor (shorter) axes
angle:      a number in degrees indicating the rotation of the ellipse around its center point,
            0 aligns the major axis horizontally
startAngle: starting at one end of major axis, indicates how far to rotate before starting to draw
            the shape, allows arcs to be drawn
endAngle:   starting at one end of major axis, indicates where the end of the filled section should be
color:      A tuple (b, g, r) for the color of the shape
thickness:  OPTIONAL, if present and positive, thickness of border line, if -1 then draws filled shape
"""


import cv2
import numpy as np

# CHANGING CENTER POSITION
for centerPos in [(250, 250), (150, 100), (300, 450), (0, 475)]:
    canvas = 255 * np.ones((500, 500, 3), np.uint8)
    cv2.putText(canvas,
                "Changing center point: " + str(centerPos),
                (15, 30),
                cv2.FONT_HERSHEY_PLAIN,
                0.9,
                (0, 0, 0))
    cv2.ellipse(canvas, centerPos, (100, 50), 0, 0, 360, (20, 0, 150), -1)

    cv2.imshow("Ellipses", canvas)
    cv2.waitKey()

# CHANGING SIZE
for axis1 in range(50, 251, 100):
    for axis2 in range(50, 251, 100):
        canvas = 255 * np.ones((500, 500, 3), np.uint8)
        cv2.putText(canvas,
                    "Changing size: " + str((axis1, axis2)),
                    (15, 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.9,
                    (0, 0, 0))
        cv2.ellipse(canvas, (250, 250), (axis1, axis2), 0, 0, 360, (150, 20, 0), -1)

        cv2.imshow("Ellipses", canvas)
        cv2.waitKey()

# CHANGING ANGLE
for angle in range(0, 181, 30):
    canvas = 255 * np.ones((500, 500, 3), np.uint8)
    cv2.putText(canvas,
                "Changing angle of rotation: " + str(angle),
                (15, 30),
                cv2.FONT_HERSHEY_PLAIN,
                0.9,
                (0, 0, 0))
    cv2.ellipse(canvas, (250, 250), (100, 50), angle, 0, 360, (20, 150, 0), -1)

    cv2.imshow("Ellipses", canvas)
    cv2.waitKey()

# CHANGING STARTING ANGLE
for angle in [0, 45]:
    for startAng in range(0, 350, 30):
        canvas = 255 * np.ones((500, 500, 3), np.uint8)
        cv2.putText(canvas,
                    "Changing angle " + str(angle) + " and starting angle: " + str(startAng),
                    (15, 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.9,
                    (0, 0, 0))
        cv2.ellipse(canvas, (250, 250), (100, 50), angle, startAng, 360, (20, 150, 150), -1)

        cv2.imshow("Ellipses", canvas)
        cv2.waitKey()

# CHANGING STARTING ANGLE AND ENDING ANGLE
for startAng in [0, 45, 90]:
    for endAng in range(30, 361, 30):
        canvas = 255 * np.ones((500, 500, 3), np.uint8)
        cv2.putText(canvas,
                    "Changing starting angle " + str(startAng) + " and ending angle: " + str(endAng),
                    (15, 30),
                    cv2.FONT_HERSHEY_PLAIN,
                    0.9,
                    (0, 0, 0))
        cv2.ellipse(canvas, (250, 250), (100, 50), 0, startAng, endAng, (150, 20, 150), -1)

        cv2.imshow("Ellipses", canvas)
        cv2.waitKey()


cv2.destroyAllWindows()


