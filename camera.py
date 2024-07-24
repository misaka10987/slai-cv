#!/usr/local/bin/python3

import cv2 as cv
import numpy as np
from numpy import ndarray
from typing import List, Sequence, Tuple, Optional, Any, Callable


class Image:
    img: ndarray = None
    color: str = "bgr"

    def __init__(self, img: ndarray, color: str = "bgr"):
        self.img = img
        self.color = color

    def mat(self) -> ndarray:
        return self.img.copy()

    def display(self, name: str):
        cv.imshow(name, self.img)

    def clone(self) -> "Image":
        return Image(self.img.copy(), color=self.color)

    def blur(self, k_size: int) -> "Image":
        return Image(cv.GaussianBlur(self.img, (k_size, k_size), 0), color=self.color)

    def gray(self) -> "Image":
        if self.color == "gray":
            return self
        return Image(cv.cvtColor(self.img, cv.COLOR_BGR2GRAY), color="gray")

    def hsv(self) -> "Image":
        if self.color == "hsv":
            return self
        assert self.color == "bgr"
        return Image(cv.cvtColor(self.img, cv.COLOR_BGR2HSV), color="hsv")

    def bgr(self) -> "Image":
        if self.color == "bgr":
            return self
        assert self.color == "hsv"
        return Image(cv.cvtColor(self.img, cv.COLOR_HSV2BGR))

    def canny(self, lo: int, hi: int) -> "Image":
        assert self.color == "gray"
        return Image(cv.Canny(self.img, lo, hi), color="gray")

    def open(self, k_size: int) -> "Image":
        img = cv.morphologyEx(self.img, cv.MORPH_OPEN,
                              cv.getStructuringElement(cv.MORPH_OPEN, (k_size * 2 + 1, k_size * 2 + 1))
                              )
        return Image(img, color=self.color)

    def mirror(self) -> "Image":
        img = self.img[:, ::-1]
        return Image(img, color=self.color)

    def in_range(self, lo: ndarray, hi: ndarray) -> "Image":
        img = cv.inRange(self.img, lo, hi)
        return Image(img, color="gray")

    def find_contour(self, mode, method) -> Sequence[ndarray]:
        con, _ = cv.findContours(self.img, mode, method)
        return con

    def abs_diff(self, other: "Image") -> "Image":
        assert self.color == other.color
        img = cv.absdiff(self.img, other.img)
        return Image(img, color=self.color)

    def draw_rect(self, begin: (int, int), end: (int, int), color: (int, int, int) = (255, 0, 255)) -> "Image":
        img = self.img.copy()
        cv.rectangle(img, begin, end, color)
        return Image(img, color=self.color)


class Camera:
    cap = cv.VideoCapture(0)
    prev: Image = None
    curr: Image = None
    blur: int = 0
    mirror: bool = True
    gray: bool = False
    open: int = 0
    edge: bool = False
    red: bool = True
    green: bool = True
    blue: bool = True
    contour: bool = False
    selecting: bool = False

    def __init__(self):
        self.prev = self.__next__()

    def __iter__(self):
        return self

    def __next__(self) -> Image:
        ipt = cv.waitKey(10)
        ipt = chr(ipt & 0xFF)
        match ipt:
            case '\x1b' | 'q' | 'Q':
                raise StopIteration
            case ' ':
                self.gray = not self.gray
            case 'm':
                self.mirror = not self.mirror
            case 'u':
                self.blur += 1
            case 'U':
                if self.blur > 0:
                    self.blur -= 1
            case 'o':
                self.open += 1
            case 'O':
                if self.open > 0:
                    self.open -= 1
            case 'e':
                self.edge = not self.edge
            case '/':
                self.selecting = not self.selecting
        _, img = self.cap.read()
        img = Image(img).blur(self.blur * 2 + 1)
        if self.gray:
            img = img.gray()
        img = img.open(self.open)
        if self.edge:
            img = img.canny(100, 200)
        if self.mirror:
            img = img.mirror()
        if self.selecting and self.curr_select() is not None:
            p_0, p_1 = self.curr_select()
            img = img.draw_rect(p_0, p_1)
        self.prev = self.curr
        self.curr = img
        return self.curr

    def diff(self) -> "Image":
        return self.curr.abs_diff(self.prev)

    cursor: (int, int) = None
    __drag_start: (int, int) = None
    __dragging: bool = False
    __selected: ((int, int), (int, int)) = None

    def curr_select(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        if self.__drag_start is not None and self.cursor is not None and self.__dragging:
            return self.__drag_start, self.cursor
        return None

    def drag_select(self):
        def handler(event, x, y, _flags, _param):
            self.cursor = (x, y)
            if not self.selecting:
                return None
            h, w = self.curr.img.shape[:2]
            match event:
                case cv.EVENT_LBUTTONDOWN:
                    self.__selected = None
                    self.__drag_start = x, y
                    self.__dragging = True
                case cv.EVENT_LBUTTONUP:
                    self.__dragging = False
                    xo, yo = self.__drag_start  # first compute upperleft anbd lower right
                    x0 = max(0, min(xo, x))
                    x1 = min(w, max(xo, x))
                    y0 = max(0, min(yo, y))
                    y1 = min(h, max(yo, y))
                    self.__selected = ((x0, y0), (x1, y1))

        return handler

    def selected(self) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        if self.__selected is None:
            return None
        s = self.__selected
        self.__selected = None
        return s


def main():
    camera = Camera()
    for frame in camera:
        cv.setMouseCallback("camera", camera.drag_select())
        s = camera.selected()
        if s is not None:
            print(s)
        frame.display("camera")


if __name__ == "__main__":
    main()
