#!/usr/local/bin/python3

import cv2 as cv


def mile1():
    import numpy as np
    img = cv.imread("./res/img/BallFinding/Blue/Blue1BG1Mid.jpg")

    cv.imshow("original", img)

    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    lower = np.array([210 / 2, 32, 32], np.uint8)
    upper = np.array([270 / 2, 216, 216], np.uint8)

    img = cv.inRange(img, lower, upper)

    cv.imshow("filtered", img)

    img = cv.Canny(img, 100, 200)

    cv.imshow("edge", img)

    con, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(img, con, -1, (0, 255, 0), 2)
    for c in con:
        ulx, uly, wid, hgt = cv.boundingRect(c)
        cv.rectangle(img, (ulx, uly), (ulx + wid, uly + hgt), (0, 0, 255), 2)
        conv = cv.convexHull((c))
        cv.drawContours(img, [conv], -1, (255, 255, 0), 1)
    cv.imshow("contours", img)
    cv.waitKey(0)


def mile2():
    from camera import Camera
    camera = Camera()
    p = camera.cap.read()
    for frame in camera:
        diff = cv.absdiff(p, frame)
        img = diff
        p = frame
        con, _ = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        cv.drawContours(img, con, -1, (0, 255, 0), 2)
        for c in con:
            ulx, uly, wid, hgt = cv.boundingRect(c)
            cv.rectangle(img, (ulx, uly), (ulx + wid, uly + hgt), (0, 0, 255), 2)
            conv = cv.convexHull((c))
            cv.drawContours(img, [conv], -1, (255, 255, 0), 1)
        cv.imshow("contours", img)
        cv.imshow("diff", diff)


def mile3():
    from camera import Camera
    import numpy as np

    def show_hist(h):
        """Takes in the histogram, and displays it in the hist window."""
        bin_count = h.shape[0]
        bin_w = 24
        img = np.zeros((256, bin_count * bin_w, 3), np.uint8)
        for i in range(bin_count):
            h = int(h[i])
            cv.rectangle(img, (i * bin_w + 2, 255), ((i + 1) * bin_w - 2, 255 - h),
                         (int(180.0 * i / bin_count), 255, 255),
                         -1)
        img = cv.cvtColor(img, cv.COLOR_HSV2BGR)
        cv.imshow('hist', img)

    camera = Camera()
    for frame in camera:
        hsv = frame.hsv()
        mask = hsv.in_range(np.array((0, 60, 32)), np.array((180, 255, 255)))
        select = camera.selected()
        cv.setMouseCallback("camshift", camera.drag_select())
        if select is not None:
            print(select)
            (x_0, y_0), (x_1, y_1) = select
            track_window = (x_0, y_0, x_1 - x_0, y_1 - y_0)
            hsv_roi = hsv.img[y_0:y_1, x_0:x_1]
            mask_roi = mask.img[y_0:y_1, x_0:x_1]
            hist = cv.calcHist([hsv_roi], [0], mask_roi, [16], [0, 180])
            cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
            hist = hist.reshape(-1)
            show_hist(hist)
            vis_roi = frame.img[y_0:y_1, x_0:x_1]
            cv.bitwise_not(vis_roi, vis_roi)
            frame.img[mask == 0] = 0
        frame.display("camshift")


def main():
    mile3()


if __name__ == "__main__":
    main()
