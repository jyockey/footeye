import numpy as np
import cv2 as cv

THRESHOLD = 120
MIN_SCENE = 10

cap = cv.VideoCapture('c:\\proj\\footeye\\highlight_vid.mp4')

last_hsv = None
frame_idx = 0
scene_idx = 0
while cap.isOpened():
    ret, frame = cap.read()
    frame_idx = frame_idx + 1
    scene_idx = scene_idx + 1
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    frame_hsv = cv.split(cv.cvtColor(frame, cv.COLOR_BGR2HSV))

    isNewScene = False
    if last_hsv is None:
        isNewScene = True
    elif scene_idx > MIN_SCENE:
        num_pixels = frame_hsv[0].shape[0] * frame_hsv[0].shape[1]
        sumDiff = 0
        for i in range(3):
            sumDiff = sumDiff + np.sum(np.abs(frame_hsv[i] - last_hsv[i]))
        avgDiff = sumDiff / (3 * float(num_pixels))
        cv.putText(frame, str(avgDiff), (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if avgDiff > THRESHOLD:
            isNewScene = True
            print(str(avgDiff))

    cv.imshow('frame', frame)
    if cv.waitKey(0 if isNewScene else 1) == ord('q'):
        break

    last_hsv = frame_hsv

    if isNewScene:
        print("New scene at frame " + str(frame_idx))
        scene_idx = 0

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()
