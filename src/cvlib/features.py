import cv2 as cv
import numpy as np

import frameutils

lower_green = np.array([30, 20, 20])
upper_green = np.array([90, 180, 220])


def mask_green(frame):
    return frameutils.mask_color_range(frame, lower_green, upper_green)


def mask_white(frame):
    grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, frame = cv.threshold(grayImage, 185, 255, cv.THRESH_BINARY)
    return frame


def pitch_mask(frame):
    blurred = cv.medianBlur(frame, 5)
    mask = mask_green(blurred)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=3)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    return mask


def on_field_mask(pitchMask):
    contours, hierarchy = cv.findContours(
            pitchMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largestContour = max(contours, key=cv.contourArea)
    hull = cv.convexHull(largestContour)
    blank = np.zeros(pitchMask.shape[0:2], dtype="uint8")
    return cv.drawContours(
            blank, [hull], -1, (255, 255, 255), -1)


def mask_to_field(frame):
    return cv.bitwise_and(frame, frame, mask=on_field_mask(pitch_mask(frame)))


def extract_players(frame):
    pitchMask = pitch_mask(frame)
    frame = cv.bitwise_and(frame, frame, mask=on_field_mask(pitchMask))
    reversedMask = cv.bitwise_not(pitchMask)
    players = cv.bitwise_and(frame, frame, mask=reversedMask)
    return players


def find_lines(frame):
    # find edges
    cv.imshow('frame', frame)
    cv.waitKey(0)
    mask = mask_white(frame)
    cv.imshow('frame', mask)
    cv.waitKey(0)
    lines = cv.HoughLinesP(mask, 1, np.pi / 180, 50, None, 100, 40)
    if lines is not None:
        for i in range(0, len(lines)):
            li = lines[i][0]
            cv.line(frame, (li[0], li[1]), (li[2], li[3]), (50, 50, 255),
                    3, cv.LINE_AA)
    cv.imshow('frame', frame)
    cv.waitKey(0)
    return frame


def interactive_find_pitch(frame):
    blurred = cv.medianBlur(frame, 3)
    mask = mask_green(blurred)
    kernel = np.ones((5, 5), np.uint8)
    morphed = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=5)
    morphed = cv.morphologyEx(morphed, cv.MORPH_CLOSE, kernel, iterations=1)
    contours, hierarchy = cv.findContours(
            morphed, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largestContour = max(contours, key=cv.contourArea)
    blank = np.zeros((frame.shape[0], frame.shape[1]), dtype="uint8")
    contour_mask = cv.drawContours(
            blank, [largestContour], -1, (255, 255, 255), -1)
    masked = cv.bitwise_and(frame, frame, mask=contour_mask)
    all_frames = [frame, blurred, mask, morphed, contour_mask, masked]
    idx = 0
    while (idx < len(all_frames)):
        cv.imshow('frame', all_frames[idx])
        key = cv.waitKeyEx(0)
        if key == 2424832:
            idx = idx - 1
        else:
            idx = idx + 1
    return masked
