import footeye.cvlib.frameutils as frameutils
import footeye.cvlib.features as features

import cv2 as cv


features.enable_logging()
frame = frameutils.extract_frame('c:\\stuff\\portland_la.ts', 75950)

features.extract_players(frame)
idx = 0
key = 0
while (key != 113):
    cv.imshow('frame', features.logframes[idx])
    key = cv.waitKeyEx(0)
    if key == 2424832:
        idx = idx - 1
    else:
        idx = (idx + 1) % len(features.logframes)

#pitch = features.interactive_find_pitch(frame)
#edges = features.find_lines(pitch)
#cv.imshow('frame', edges)
#cv.imshow('frame', features.extract_players(frame))
#cv.waitKey(0)

#vid = cv.VideoCapture('c:\\proj\\footeye\\king_vid.mp4')
#while vid.isOpened():
#    ret, frame = vid.read()
#    # if frame is read correctly ret is True
#    if not ret:
#        print("Can't receive frame (stream end?). Exiting ...")
#        break
#    cv.imshow('frame', features.extract_players(frame))
#    if cv.waitKey(1) == ord('q'):
#        break
# When everything done, release the capture
#vid.release()
#cv.destroyAllWindows()
