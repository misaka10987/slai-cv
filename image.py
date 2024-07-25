import copy

import cv2 as cv
from numpy import ndarray
from typing import Sequence, Set, Dict, List, Tuple
from itertools import chain

Point = Tuple[int, int]

Rectangle = Tuple[Point, Point]

Color = Tuple[int, int, int]


class Image:
    img: ndarray = None
    color: str = "bgr"
    rect: List[Tuple[Rectangle, Color]] = []
    box: Dict[str, List[Rectangle]] = {}
    contour: List[ndarray] = []

    def __init__(self, img: ndarray, color: str = "bgr", box=None, contour=None):
        self.img = img
        self.color = color
        if box:
            self.box = box
        if contour:
            self.contour = contour

    def mat(self) -> ndarray:
        return self.img.copy()

    def display(self, name: str) -> "Image":
        img = self.img.copy()
        for (begin, end), color in self.rect:
            cv.rectangle(img, [*begin], [*end], color)
        cv.imshow(name, self.img)
        return self

    def clone(self) -> "Image":
        return copy.copy(self)

    def map(self, f) -> "Image":
        return f(self)

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

    def affine(self, mat: ndarray, size: (int, int) = None) -> "Image":
        return Image(cv.warpAffine(self.img, mat, size))

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

    def find_contour(self, mode, method) -> "Image":
        img = self.clone()
        con, _ = cv.findContours(self.img, mode, method)
        img.contour += [c for c in con]
        return img

    def abs_diff(self, other: "Image") -> "Image":
        assert self.color == other.color
        img = cv.absdiff(self.img, other.img)
        return Image(img, color=self.color)

    def draw_rect(self, rect: Rectangle, color: Color = (255, 0, 255)) -> "Image":
        self.rect += [(rect, color)]
        return self

    def plot_box(self, name: Set[str] = None):
        img = self.clone()
        try:
            it = chain(*[self.box[n] for n in name])
        except TypeError:
            it = chain(*self.box.values())
        for a, b in it:
            img = img.draw_rect(a, b)
        return img


class CascadeFinder:
    name: str

    def __init__(self, name: str, path: str):
        self.name = name
        self.cascade = cv.CascadeClassifier(path)

    def __call__(self, img: Image) -> Image:
        img = img.clone()
        rect = [((x, y), ((x + w), (y + h))) for x, y, w, h in self.cascade.detectMultiScale(img.gray().mat(), 1.3, 5)]
        img.box[self.name] = rect
        return img
