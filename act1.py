#!/usr/local/bin/python3

import subprocess
import cv2 as cv
import random


def slideshow(path: str):
    img = subprocess.check_output(f'ls {path}/*.{{jpg,png}}', shell=True, text=True).split('\n')[:-2]
    for i in img:
        i = cv.imread(i)
        cv.imshow("Slideshow", i)
        cv.waitKey()


def shuffle(path: str):
    img = cv.imread(path)
    channel = list(cv.split(img))
    random.shuffle(channel)
    img = cv.merge(channel)
    cv.imshow("Shuffled", img)
    cv.waitKey()


def snowleo():
    img = cv.imread("./res/img/snowLeo2.jpg")
    cv.circle(img, (150, 150), 100, (0, 0, 255), 10)
    cv.imshow("SnowLeo", img)
    cv.waitKey()


def main():
    slideshow("./res/img")
    shuffle("./res/img/chicago.jpg")
    snowleo()


if __name__ == "__main__":
    main()
