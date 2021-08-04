import cv2 as cv
import footeye.cvlib.frameutils as frameutils
import footeye.cvlib.features as features
import footeye.model.vidinfo as vidinfo
import numpy as np
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
# sample 20 frames
frameIds = [np.random.randint(vid.frameCount) for x in range(20)]
frames = frameutils.extract_frames(vid.vidFilePath, frameIds)
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)

print(medianFrame.shape)
features.log_frame(medianFrame, "Median")
variance = np.var(frames, axis=0)
flattened = variance.reshape(-1, variance.shape[-1])
print(flattened.shape)
summed = np.sum(flattened, axis=1)
interpolated = np.interp(summed, (summed.min(), summed.max()), (255, 0))
greyscale = interpolated.reshape((medianFrame.shape[0], -1)).astype(dtype=np.uint8)
features.log_frame(greyscale, "Variance grayscale")
_, variance_mask = cv.threshold(greyscale, 220, 255, cv.THRESH_BINARY)
features.log_frame(variance_mask, "Variance Mask")
masked = cv.bitwise_and(medianFrame, medianFrame, mask=variance_mask)
features.log_frame(masked, "Masked")

#frameutils.pixel_hue_variance(frames)
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
