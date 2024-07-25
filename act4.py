#!/usr/local/bin/python3


from image import CascadeFinder
from camera import Camera

finder = CascadeFinder("data/haar/haarcascade_frontalface_alt0.xml")

camera = Camera()

for frame in camera:
    frame.map(finder).display("found")
