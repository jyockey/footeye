from ..cvlib import frameutils
from ..cvlib import features

import cv2 as cv


frame = frameutils.extract_frame('c:\\stuff\\portland_la.ts', 75950)
frame = features.interactive_find_pitch(frame)
edges = features.find_lines(frame)
cv.imshow('frame', edges)
cv.waitKey(0)

vid = cv.VideoCapture('c:\\stuff\\portland_la.ts')
while vid.isOpened():
    ret, frame = vid.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    cv.imshow('frame', features.extract_players(frame))
    if cv.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
vid.release()
cv.destroyAllWindows()
