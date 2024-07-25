#!/usr/local/bin/python3

import cv2 as cv
from typing import Tuple, Optional

from image import Image


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
