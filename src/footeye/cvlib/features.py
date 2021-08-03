from sklearn.cluster import MeanShift, estimate_bandwidth

import cv2 as cv
import numpy as np
import footeye.cvlib.frameutils as frameutils

COL_RED = (0, 0, 255)


lower_green = np.array([30, 40, 40])
upper_green = np.array([90, 180, 220])

logframes = None


def enable_logging():
    global logframes
    logframes = []


def log_frame(frame, desc):
    global logframes
    if logframes is not None:
        if (len(frame.shape) < 3):
            copy = cv.cvtColor(frame, cv.COLOR_GRAY2BGR)
        else:
            copy = frame.copy()
        cv.putText(copy, desc, (20, 20), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                   (0, 0, 255), 2)
        logframes.append(copy)


def mask_green(frame):
    return frameutils.mask_color_range(frame, lower_green, upper_green)


def mask_white(frame):
    grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, frame = cv.threshold(grayImage, 185, 255, cv.THRESH_BINARY)
    return frame


def pitch_mask(frame):
    frame = cv.medianBlur(frame, 3)
    log_frame(frame, "Blurred")
    mask = mask_green(frame)
    log_frame(mask, "Green Mask")
    kernel = np.ones((4, 4), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=3)
    log_frame(mask, "Morphed 1")
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    log_frame(mask, "Morphed 2")
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
    fieldMask = on_field_mask(pitch_mask(frame))
    log_frame(fieldMask, "Field Mask")
    return cv.bitwise_and(frame, frame, mask=fieldMask)


def field_not_pitch_mask(frame):
    pitchMask = pitch_mask(frame)
    onFieldMask = on_field_mask(pitchMask)
    field = cv.bitwise_and(frame, frame, mask=onFieldMask)
    log_frame(field, "Field")
    notPitchMask = cv.bitwise_and(
        onFieldMask, onFieldMask, mask=cv.bitwise_not(pitchMask))
    log_frame(notPitchMask, "Not pitch mask")
    return notPitchMask
    fieldNotPitch = cv.bitwise_and(field, field, mask=notPitchMask)
    log_frame(fieldNotPitch, "Field not pitch")
    return fieldNotPitch


def _likely_player(contour):
    x, y, w, h = cv.boundingRect(contour)
    aspect = w / h
    return aspect < 8 and aspect > 0.125 and h > 20


def extract_players(frame):
    fieldNotPitch = field_not_pitch_mask(frame)
    contours, hierarchy = cv.findContours(
        fieldNotPitch, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rectFrame = frame.copy()
    drawn = cv.drawContours(frame, contours, -1, COL_RED, 3)
    log_frame(drawn, "allContours")
    contours = filter(_likely_player, contours)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(rectFrame, (x,y), (x+w,y+h), COL_RED, 3)
    log_frame(rectFrame, "boundingRects")


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
