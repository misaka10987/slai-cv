#!/usr/local/bin/python3

import cv2 as cv


def mile1():
    from camera import Camera
    camera = Camera()
    for frame in camera:
        img = frame
        img = img[:, ::-1, :]
        cv.imshow("Webcam", img)
        ipt = cv.waitKey(10)
        ipt = chr(ipt & 0xFF)
    cv.destroyAllWindows()


def mile2():
    from camera import Camera
    angle = 0
    camera = Camera()
    for frame in camera:
        (row, col, _) = frame.shape
        rot = cv.getRotationMatrix2D((col / 2, row / 2), angle, 1)
        img = cv.warpAffine(frame, rot, (col, row))
        angle += 1
        cv.imshow("Webcam", img)
    cv.destroyAllWindows()


def mile3():
    from cmath import sin
    from time import time
    from camera import Camera
    camera = Camera()
    for frame in camera:
        k_size = int(30 + 30 * sin(time() / 20).real) // 2 * 2 + 1
        cv.imshow("cam", frame)
        frame = cv.GaussianBlur(frame, (k_size, k_size), 0)
        cv.imshow("Blurred", frame)


def mile4():
    import numpy as np
    from camera import Camera
    camera = Camera()
    for frame in camera:
        frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

        lower = np.array([75 / 2, 32, 32], np.uint8)
        upper = np.array([135 / 2, 216, 216], np.uint8)

        mask = cv.inRange(frame, lower, upper)
        cv.imshow("threshold", mask)


def main():
    mile4()


if __name__ == "__main__":
    main()
