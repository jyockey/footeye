import cv2 as cv
import numpy as np


def extract_frame(file, frame_idx):
    vid = cv.VideoCapture(file)
    vid.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = vid.read()
    vid.release()
    return frame


def mask_color_range(frame, lower_color, upper_color):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, lower_color, upper_color)
