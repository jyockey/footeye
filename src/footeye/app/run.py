import cv2 as cv
import footeye.cvlib.frameutils as frameutils
import footeye.cvlib.features as features
import footeye.model.vidinfo as vidinfo
import os.path


def load_video(path):
    if not os.path.isfile(path):
        raise "No such file: " + path
    vidInfo = vidinfo.VidInfo(path)
    vid = cv.VideoCapture(path)
    frameCount = vid.get(cv.CAP_PROP_FRAME_COUNT)
    vid.release()
    vidInfo.frameCount = frameCount
    return vidInfo


features.enable_logging()
vid = load_video('c:\\stuff\\portland_la.ts')
frames = frameutils.extract_frames(vid.vidFilePath, [10000, 75000, 100000])
frameutils.pixel_hue_variance(frames)
# frame = frameutils.extract_frame(vid.vidFilePath, 75950)
# print(vid.frameCount)

# features.extract_players(frame)

idx = 0
key = 0
while len(features.logframes) > 0 and key != 113:
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
