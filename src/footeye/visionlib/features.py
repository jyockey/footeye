from sklearn.cluster import MeanShift, estimate_bandwidth

import cv2 as cv
import numpy as np
import footeye.visionlib.frameutils as frameutils
import footeye.utils.framedebug as framedebug

SCENE_DIFF_THRESHOLD = 80

FRAME_GREEN_RATIO_CUTOFF = 0.45

COL_RED = (0, 0, 255)
COL_WHITE = (255, 255, 255)
COL_YELLOW = (0, 255, 255)


def mask_white(frame):
    grayImage = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    _, frame = cv.threshold(grayImage, 150, 255, cv.THRESH_BINARY)
    return frame


def colorclick(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN:
        frame = param
        print(frame[y][x])


def colorpick_frame(frame):
    framedebug.show_for_click(
            frame,
            colorclick,
            cv.cvtColor(frame, cv.COLOR_BGR2HSV))


def pitch_mask(frame, min_pitch_color, max_pitch_color):
    frame = cv.medianBlur(frame, 3)
    framedebug.log_frame(frame, "Blurred")
    # framedebug.show_for_click(
    # frame, colorclick, cv.cvtColor(frame, cv.COLOR_BGR2HSV))
    mask = frameutils.mask_color_range(frame, min_pitch_color, max_pitch_color)
    framedebug.log_frame(mask, "Green Mask")

    # If there's insufficient green in a frame, we assume it isn't valid for
    # pitch location.
    # TODO: This should probably be done at the on_field_mask layer instead
    pixels = cv.countNonZero(mask)
    frame_area = frame.shape[0] * frame.shape[1]
    if (pixels / frame_area) < FRAME_GREEN_RATIO_CUTOFF:
        return None

    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((8, 8), np.uint8), iterations=3)
    framedebug.log_frame(mask, "Morphed 1")
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((20, 20), np.uint8), iterations=3)
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
    if pitchMask is None:
        return frame
    fieldMask = on_field_mask(pitchMask)
    framedebug.log_frame(fieldMask, "Field Mask")
    return cv.bitwise_and(frame, frame, mask=fieldMask)


def field_not_pitch_mask(frame, pitchMask, vidinfo):
    onFieldMask = on_field_mask(pitchMask)
    framedebug.log_frame(onFieldMask, "On Field")
    field = cv.bitwise_and(frame, frame, mask=onFieldMask)
    framedebug.log_frame(field, "Field")
    #equalized = frameutils.clahe_frame(field)
    #framedebug.log_frame(equalized, "Equalized")
    equalized = field
    mask = frameutils.mask_color_range(equalized, vidinfo.fieldColorExtents[0], vidinfo.fieldColorExtents[1])
    framedebug.log_frame(mask, "Green Mask")
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=3)
    framedebug.log_frame(mask, "Morphed 1")
    notPitchMask = cv.bitwise_and(
        onFieldMask, onFieldMask, mask=cv.bitwise_not(mask))
    framedebug.log_frame(notPitchMask, "Not pitch mask")
    return notPitchMask
    # fieldNotPitch = cv.bitwise_and(field, field, mask=notPitchMask)
    # framedebug.log_frame(fieldNotPitch, "Field not pitch")
    # return fieldNotPitch


def _is_similar_rect(rect, reference_rect):
    wratio = rect[2] / reference_rect[2]
    hratio = rect[3] / reference_rect[3]
    return (0.5 < wratio < 2.0) and (0.5 < hratio < 2.0)


def extract_feature_rects(frame, vidinfo):
    pitchMask = pitch_mask(
      frame, vidinfo.fieldColorExtents[0], vidinfo.fieldColorExtents[1])
    if pitchMask is None:
        frameutils.header_text(frame, 'NOT PITCH')
        framedebug.log_frame(frame, "")
        return []
    fieldNotPitch = field_not_pitch_mask(frame, pitchMask, vidinfo)
    contours, hierarchy = cv.findContours(
        fieldNotPitch, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if (framedebug.is_enabled()):
        drawn = cv.drawContours(frame, contours, -1, COL_RED, 3)
        framedebug.log_frame(drawn, "allContours")
    rects = list(map(lambda c: list(cv.boundingRect(c)), contours))
    return list(filter(lambda r: r[2] > 10 and r[3] > 10, rects))


def estimate_player_size(rects):
    return np.median(rects, axis=0)


def extract_players_from_rects(frame, rects, player_size):
    playerRects = filter(lambda c: _is_similar_rect(c, player_size), rects)
    rectFrame = frame.copy()
    if (framedebug.is_enabled()):
        frameutils.draw_rect(rectFrame, player_size.astype(int), COL_WHITE, 3)
        for rect in rects:
            frameutils.draw_rect(rectFrame, rect, COL_RED, 3)
        framedebug.log_frame(rectFrame, "boundingRects")
    for rect in playerRects:
        frameutils.draw_rect(frame, rect, COL_YELLOW, 3)
    framedebug.log_frame(frame, "playerRects")
    return frame


def extract_players(frame, vidinfo):
    rects = extract_feature_rects(frame, vidinfo)
    if not rects:
        return frame
    player_size = estimate_player_size(rects)
    return extract_players_from_rects(frame, rects, player_size)


def pitch_orientation(frame, vidinfo):
    onFieldMask = pitch_mask(
      frame, vidinfo.fieldColorExtents[0], vidinfo.fieldColorExtents[1])
    field = cv.bitwise_and(frame, frame, mask=onFieldMask)
    framedebug.log_frame(field, "Field")
    # equalized = frameutils.clahe_frame(field)
    # framedebug.log_frame(equalized, "Equalized")
    white_field = mask_white(field)
    framedebug.log_frame(white_field, 'Mask White')
    pitch = cv.imread('pitch.png', 0)
    orb = cv.SIFT_create()
    kpP, desP = orb.detectAndCompute(pitch, None)
    kpW, desW = orb.detectAndCompute(white_field, None)
    kpImgP = cv.drawKeypoints(pitch, kpP, None, color=(0, 255, 0), flags=0)
    kpImgW = cv.drawKeypoints(white_field, kpW, None, color=(0, 255, 0), flags=0)
    framedebug.log_frame(kpImgP, "Key Points")
    framedebug.log_frame(kpImgW, "Key Points 2")

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desP, desW, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)
    print(good)

    src_pts = np.float32([kpP[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpW[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()
    h, w = pitch.shape
    pts = np.float32([[0,0],[0,h-1],[w-1,h-1],[w-1,0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)
    img2 = cv.polylines(white_field ,[np.int32(dst)], True, 255, 3, cv.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv.drawMatches(pitch, kpP, white_field, kpW, good, None, **draw_params)
    framedebug.log_frame(img3, 'wow')


def find_lines(frame, vidinfo):
    onFieldMask = pitch_mask(
      frame, vidinfo.fieldColorExtents[0], vidinfo.fieldColorExtents[1])
    if (onFieldMask is None):
        return frame
    field = cv.bitwise_and(frame, frame, mask=onFieldMask)
    framedebug.log_frame(field, "Field")
    # equalized = frameutils.clahe_frame(field)
    # framedebug.log_frame(equalized, "Equalized")
    mask = mask_white(field)
    framedebug.log_frame(mask, 'Mask White')
    linesFrame = frame.copy()
    lines = cv.HoughLinesP(mask, 1, np.pi / 180, 50, None, frame.shape[1] / 6, 20)
    if lines is not None:
        for i in range(0, len(lines)):
            li = lines[i][0]
            cv.line(linesFrame, (li[0], li[1]), (li[2], li[3]), (50, 50, 255),
                    1, cv.LINE_AA)
    framedebug.log_frame(linesFrame, 'lines')
    return linesFrame


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
      np.array([min(255, int(mean + stdev)), 255, 200])]


def scene_break_check(frame, prev_frame):
    if prev_frame is None:
        return True

    size = frame[0].shape[0] * frame[0].shape[1]
    buckets = 100

    hist = cv.calcHist([frame], [0], None, [buckets], [0, 256])
    prev_hist = cv.calcHist([prev_frame], [0], None, [buckets], [0, 256])

    diff = np.sum(np.abs(hist - prev_hist)) / size
    return diff > SCENE_DIFF_THRESHOLD
