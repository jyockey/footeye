import cv2 as cv
import numpy as np

from matplotlib import pyplot as plt


def extract_frame(file, frame_idx):
    return extract_frames(file, [frame_idx])[0]


def extract_frames(file, frame_idxs):
    frames = []
    vid = cv.VideoCapture(file)
    for idx in frame_idxs:
        vid.set(cv.CAP_PROP_POS_FRAMES, idx)
        ret, frame = vid.read()
        if frame is None:
            print("Got no frame for idx " + str(idx))
        else:
            frames.append(frame)
    vid.release()
    return frames


def mask_color_range(frame, lower_color, upper_color):
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, lower_color, upper_color)


def draw_rect(frame, rect, color, width):
    x, y, w, h = rect
    cv.rectangle(frame, (x, y), (x + w, y + h), color, width)
    

# Because hue values in HSV are circular, where value 0 and value 179 (max) are
# very near rather than very far, calculating the (shortest-distance)
# difference between two hues requires some special math
def hue_diff(hueval1, hueval2):
    rawval = hueval2 - hueval1
    if rawval < -90:
        return 180 + rawval
    elif rawval > 90:
        return -(180 - rawval)
    else:
        return rawval


def running_mean_hue(existing_mean, new_val, source_count):
    diff = hue_diff(existing_mean, new_val)
    return (existing_mean + (diff / (source_count + 1))) % 180


def pixel_mean_hues(frames):
    means = np.zeros(frames[0].shape[0] * frames[0].shape[1])
    for frameid in range(len(frames)):
        print("Frame " + str(frameid))
        frame = cv.cvtColor(frames[frameid], cv.COLOR_BGR2HSV)
        hues = frame.reshape(-1, frame.shape[-1])[:, 0]
        print(hues)
        means = np.array([running_mean_hue(cur, hues[idx], frameid)
                          for (idx, cur) in enumerate(means)])
    return means


def pixel_hue_variance(frames):
    variances = np.zeros(frames[0].shape[0] * frames[0].shape[1])
    means = pixel_mean_hues(frames)
    for frameid in range(len(frames)):
        print("Frame " + str(frameid))
        frame = cv.cvtColor(frames[frameid], cv.COLOR_BGR2HSV)
        hues = frame.reshape(-1, frame.shape[-1])[:, 0]
        print(hues)
        diffs = np.absolute(hues - means)
        variances = variances + diffs
    print(variances)


def histo(frame):
    for channel in [0]:
        print(channel)
        hist = cv.calcHist([frame], [channel], None, [256], [0, 256])
        plt.plot(hist)
        plt.xlim([0, 256])
    plt.show()
