import cv2 as cv

logframes = None


def is_enabled():
    global logframes
    return logframes is not None


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


def show_for_click(frame, callback, params):
    cv.imshow('frame', frame)
    cv.setMouseCallback('frame', callback, params)
    cv.waitKey(0)


def show_frames():
    global logframes
    idx = 0
    key = 0
    while len(logframes) > 0 and key != 113:
        cv.imshow('frame', logframes[idx])
        key = cv.waitKeyEx(0)
        if key == 2424832:
            idx = idx - 1
        else:
            idx = (idx + 1) % len(logframes)
    cv.destroyAllWindows()
