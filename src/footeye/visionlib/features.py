from sklearn.cluster import MeanShift, estimate_bandwidth

import cv2 as cv
import math
import numpy as np
import footeye.visionlib.frameutils as frameutils
import footeye.utils.framedebug as framedebug

SCENE_DIFF_THRESHOLD = 80

FRAME_GREEN_RATIO_CUTOFF = 0.45

COL_RED = (0, 0, 255)
COL_WHITE = (255, 255, 255)
COL_YELLOW = (0, 255, 255)


class Feature:
    contour = None
    rect = None
    contour_area = 0

    def __init__(self, contour):
        self.contour = contour
        self.contour_area = cv.contourArea(contour)
        self.rect = cv.boundingRect(contour)

    def width(self):
        return self.rect[2]

    def height(self):
        return self.rect[3]

    def rect_area(self):
        return self.width() * self.height()

    def aspect_ratio(self):
        return self.width() / self.height()

    def density(self):
        return self.contour_area / self.rect_area()

    def draw(self, frame, color=COL_RED):
        cv.drawContours(frame, [self.contour], -1, color, 2)
        frameutils.draw_rect(frame, self.rect, color, 1)
        text_x = self.rect[0] + self.width() + 10
        text_y = self.rect[1] + 10
        text = str(self.rect)
        cv.putText(frame, text, (text_x, text_y),
                   cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        text = ('%s / %.3g / %.3g' %
                (self.rect_area(), self.aspect_ratio(), self.density()))
        cv.putText(frame, text, (text_x, text_y + 20),
                   cv.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)


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
    equalized = frameutils.clahe_frame(field)
    framedebug.log_frame(equalized, "Equalized")
    # equalized = field
    mask = frameutils.mask_color_range(equalized, vidinfo.fieldColorExtents[0], vidinfo.fieldColorExtents[1])
    framedebug.log_frame(mask, "Green Mask")
    kernel = np.ones((3, 3), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=3)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=1)
    framedebug.log_frame(mask, "Morphed")
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


def _is_similar_area(area1, area2):
    ratio = area1 / area2
    return 0.3 < ratio < 3.0


def extract_pitch_features(frame, vidinfo):
    pitchMask = pitch_mask(
      frame, vidinfo.fieldColorExtents[0], vidinfo.fieldColorExtents[1])
    if pitchMask is None:
        frameutils.header_text(frame, 'NOT PITCH')
        framedebug.log_frame(frame, "")
        return []
    fieldNotPitch = field_not_pitch_mask(frame, pitchMask, vidinfo)
    contours, hierarchy = cv.findContours(
        fieldNotPitch, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    features = list(map(Feature, contours))
    if (framedebug.is_enabled()):
        for feature in features:
            feature.draw(frame)
        framedebug.log_frame(frame, "allContours")
    return features


def draw_player_feature_rects(frame, features, player_features, median_area):
    rectFrame = frame.copy()
    if (framedebug.is_enabled()):
        sqrt = int(math.sqrt(median_area))
        frameutils.draw_rect(rectFrame, (0, 0, sqrt, sqrt), COL_WHITE, 3)
        for feature in features:
            frameutils.draw_rect(rectFrame, feature.rect, COL_RED, 3)
        framedebug.log_frame(rectFrame, "boundingRects")
    for feature in player_features:
        frameutils.draw_rect(frame, feature.rect, COL_YELLOW, 3)
    framedebug.log_frame(frame, "playerRects")
    return frame


def _is_likely_player_feature(feature):
    aspect = feature.aspect_ratio()
    return (feature.contour_area > 20
            and feature.width() > 10 and feature.height() > 10
            and aspect > 0.25 and aspect < 4
            and feature.density() > 0.3)


def extract_players(frame, vidinfo):
    rectFrame = frame.copy()
    features = extract_pitch_features(frame, vidinfo)
    # filter out tiny or very thin features or features that do not fill a
    # majority of their bounding rect
    features = list(filter(_is_likely_player_feature, features))
    if not features:
        return frame
    median_area = np.median(
            list(map(lambda f: f.contour_area, features)))
    # print("M: %s" % median_area)
    features = list(filter(
            lambda f: _is_similar_area(median_area, f.contour_area),
            features))
    for feature in features:
        feature.draw(rectFrame, COL_WHITE)
    framedebug.log_frame(rectFrame, "playerRects")
    return rectFrame


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
