import footeye.cvlib.features as features
import footeye.cvlib.frameutils as frameutils
import footeye.model.vidinfo as vidinfo
import footeye.utils.framedebug as framedebug


framedebug.enable_logging()

vid = vidinfo.VidInfo.forPath('c:\\stuff\\portland_la.ts')

if (vid.fieldColorExtents is None):
    vid.fieldColorExtents = features.find_field_color_extents(vid)
print(vid.fieldColorExtents)
vid.save()

frame = frameutils.extract_frame(vid.vidFilePath, 75950)
features.extract_players(frame, vid)

framedebug.show_frames()

# variance = np.var(frames, axis=0)
# flattened = variance.reshape(-1, variance.shape[-1])
# print(flattened.shape)
# summed = np.sum(flattened, axis=1)
# interpolated = np.interp(summed, (summed.min(), summed.max()), (255, 0))
# greyscale = interpolated.reshape((medianFrame.shape[0], -1)).astype(dtype=np.uint8)
# features.log_frame(greyscale, "Variance grayscale")
# _, variance_mask = cv.threshold(greyscale, 220, 255, cv.THRESH_BINARY)
# features.log_frame(variance_mask, "Variance Mask")
# masked = cv.bitwise_and(medianFrame, medianFrame, mask=variance_mask)
# features.log_frame(masked, "Masked")

#frameutils.pixel_hue_variance(frames)
# frame = frameutils.extract_frame(vid.vidFilePath, 75950)
# print(vid.frameCount)

# features.extract_players(frame)


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
