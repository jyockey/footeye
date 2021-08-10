from sklearn.cluster import MeanShift, estimate_bandwidth

import cv2 as cv
import numpy as np
import footeye.visionlib.frameutils as frameutils
import footeye.utils.framedebug as framedebug


COL_RED = (0, 0, 255)


def mask_white(frame):
    grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, frame = cv.threshold(grayImage, 185, 255, cv.THRESH_BINARY)
    return frame


def colorclick(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        frame = param
        print(frame[y][x])


def pitch_mask(frame, min_pitch_color, max_pitch_color):
    frame = cv.medianBlur(frame, 3)
    framedebug.log_frame(frame, "Blurred")
    # framedebug.show_for_click(frame, colorclick, cv.cvtColor(frame, cv.COLOR_BGR2HSV))
    print(min_pitch_color)
    mask = frameutils.mask_color_range(frame, min_pitch_color, max_pitch_color)
    print(max_pitch_color)
    framedebug.log_frame(mask, "Green Mask")
    kernel = np.ones((4, 4), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=3)
    framedebug.log_frame(mask, "Morphed 1")
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=2)
    framedebug.log_frame(mask, "Morphed 2")
    return mask


def on_field_mask(pitchMask):
    contours, hierarchy = cv.findContours(
            pitchMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if (len(contours) == 0):
        return pitchMask
    largestContour = max(contours, key=cv.contourArea)
    hull = cv.convexHull(largestContour)
    blank = np.zeros(pitchMask.shape[0:2], dtype="uint8")
    return cv.drawContours(
            blank, [hull], -1, (255, 255, 255), -1)


def mask_to_field(frame):
    fieldMask = on_field_mask(pitch_mask(frame))
    framedebug.log_frame(fieldMask, "Field Mask")
    return cv.bitwise_and(frame, frame, mask=fieldMask)


def field_not_pitch_mask(frame, pitchMask):
    onFieldMask = on_field_mask(pitchMask)
    field = cv.bitwise_and(frame, frame, mask=onFieldMask)
    framedebug.log_frame(field, "Field")
    notPitchMask = cv.bitwise_and(
        onFieldMask, onFieldMask, mask=cv.bitwise_not(pitchMask))
    framedebug.log_frame(notPitchMask, "Not pitch mask")
    return notPitchMask
    fieldNotPitch = cv.bitwise_and(field, field, mask=notPitchMask)
    framedebug.log_frame(fieldNotPitch, "Field not pitch")
    return fieldNotPitch


def _likely_player(contour):
    x, y, w, h = cv.boundingRect(contour)
    aspect = w / h
    return aspect < 8 and aspect > 0.125 and h > 20


def extract_players(frame, vidinfo):
    pitchMask = pitch_mask(
      frame, vidinfo.fieldColorExtents[0], vidinfo.fieldColorExtents[1])
    fieldNotPitch = field_not_pitch_mask(frame, pitchMask)
    contours, hierarchy = cv.findContours(
        fieldNotPitch, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rectFrame = frame.copy()
    drawn = cv.drawContours(frame, contours, -1, COL_RED, 3)
    framedebug.log_frame(drawn, "allContours")
    contours = filter(_likely_player, contours)
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(rectFrame, (x,y), (x+w,y+h), COL_RED, 3)
    framedebug.log_frame(rectFrame, "boundingRects")


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


def find_field_color_extents(vid):
    # sample 20 frames
    frameIds = [np.random.randint(vid.frameCount) for x in range(20)]
    frames = frameutils.extract_frames(vid.vidFilePath, frameIds)
    medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
    framedebug.log_frame(medianFrame, "Median")
    hsvMedian = cv.cvtColor(medianFrame, cv.COLOR_BGR2HSV)
    # frameutils.histo(hsvMedian)
    hues = cv.extractChannel(hsvMedian, 0)
    stdev = np.std(hues)
    mean = np.mean(hues)
    print(stdev)
    print(mean)
    return [
      np.array([max(0, int(mean - stdev)), 0, 50]),
      np.array([min(255, int(mean + stdev)), 150, 200])]
