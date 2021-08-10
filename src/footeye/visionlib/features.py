from sklearn.cluster import MeanShift, estimate_bandwidth

import cv2 as cv
import numpy as np
import footeye.visionlib.frameutils as frameutils
import footeye.utils.framedebug as framedebug


COL_RED = (0, 0, 255)
COL_WHITE = (255, 255, 255)
COL_YELLOW = (0, 255, 255)


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
    # framedebug.show_for_click(
    # frame, colorclick, cv.cvtColor(frame, cv.COLOR_BGR2HSV))
    mask = frameutils.mask_color_range(frame, min_pitch_color, max_pitch_color)
    framedebug.log_frame(mask, "Green Mask")
    kernel = np.ones((6, 6), np.uint8)
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


def mask_to_field(frame, vidinfo):
    pitchMask = pitch_mask(
      frame, vidinfo.fieldColorExtents[0], vidinfo.fieldColorExtents[1])
    fieldMask = on_field_mask(pitchMask)
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


def _is_similar_rect(rect, reference_rect):
    wratio = rect[2] / reference_rect[2]
    hratio = rect[3] / reference_rect[3]
    return (0.5 < wratio < 2.0) and (0.5 < hratio < 2.0)


def extract_players(frame, vidinfo):
    pitchMask = pitch_mask(
      frame, vidinfo.fieldColorExtents[0], vidinfo.fieldColorExtents[1])
    fieldNotPitch = field_not_pitch_mask(frame, pitchMask)
    contours, hierarchy = cv.findContours(
        fieldNotPitch, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    rects = list(map(lambda c: list(cv.boundingRect(c)), contours))
    medians = np.median(rects, axis=0)
    playerRects = filter(lambda c: _is_similar_rect(c, medians), rects)
    rectFrame = frame.copy()
    drawn = cv.drawContours(frame, contours, -1, COL_RED, 3)
    framedebug.log_frame(drawn, "allContours")
    frameutils.draw_rect(rectFrame, medians.astype(int), COL_WHITE, 3)
    for rect in rects:
        frameutils.draw_rect(rectFrame, rect, COL_RED, 3)
    for rect in playerRects:
        frameutils.draw_rect(rectFrame, rect, COL_YELLOW, 3)
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
    return [
      np.array([max(0, int(mean - stdev)), 20, 70]),
      np.array([min(255, int(mean + stdev)), 150, 200])]
