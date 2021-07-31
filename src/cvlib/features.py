import cv2 as cv
import numpy as np

lower_green = np.array([30, 20, 20])
upper_green = np.array([90, 180, 220])


def extract_frame(file, frame_idx):
    vid = cv.VideoCapture(file)
    vid.set(cv.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = vid.read()
    vid.release()
    return frame


def mask_green(frame):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, lower_green, upper_green)


def mask_white(frame):
    grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, frame = cv.threshold(grayImage, 185, 255, cv.THRESH_BINARY)
    return frame


def pitch_mask(frame):
    blurred = cv.medianBlur(frame, 5)
    mask = mask_green(blurred)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=5)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    return mask


def on_field_mask(pitchMask):
    contours, hierarchy = cv.findContours(
            pitchMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    largestContour = max(contours, key=cv.contourArea)
    hull = cv.convexHull(largestContour)
    blank = np.zeros(frame.shape[0:2], dtype="uint8")
    return cv.drawContours(
            blank, [hull], -1, (255, 255, 255), -1)


def mask_to_field(frame):
    return cv.bitwise_and(frame, frame, mask=on_field_mask(pitch_mask(frame)))


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


# 350 450 950
#frame = extract_frame('c:\\proj\\footeye\\full_vid.mp4', 950)
#frame = interactive_find_pitch(frame)
#edges = find_lines(frame)
#cv.imshow('frame', edges)
#cv.waitKey(0)

vid = cv.VideoCapture('c:\\proj\\footeye\\highlight_vid.mp4')
while vid.isOpened():
    ret, frame = vid.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('frame', mask_to_field(frame))
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
vid.release()

#cv.destroyAllWindows()
