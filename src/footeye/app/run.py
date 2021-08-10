from footeye.app.model import VidInfo, Project

import argparse
import cv2 as cv
import pickle
import footeye.visionlib.features as features
import footeye.visionlib.frameutils as frameutils
import footeye.utils.framedebug as framedebug


def loadProject(projectFile):
    with open(projectFile, 'rb') as f:
        return pickle.load(f)


def createProjectFromVideo(videoFile):
    vid = VidInfo(videoFile)
    projName = input("Project Name?> ")
    project = Project(projName, vid)
    print('Saving to ' + project.filename())
    project.save()
    return project


def play_transformed(project, trans_function):
    vid = cv.VideoCapture(project.vidinfo.vidFilePath)
    while vid.isOpened():
        ret, frame = vid.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        cv.imshow('frame', trans_function(frame))
        if cv.waitKey(1) == ord('q'):
            break
    vid.release()


def processProject(project):
    vid = project.vidinfo
    if (vid.fieldColorExtents is None):
        vid.fieldColorExtents = features.find_field_color_extents(vid)
    project.save()

    # framedebug.enable_logging()
    # frame = frameutils.extract_frame(vid.vidFilePath, 3000)
    # features.extract_players(frame, vid)
    # framedebug.show_frames()
    play_transformed(project, lambda f: features.mask_to_field(f, vid))


def runApp():
    argparser = argparse.ArgumentParser(description='Run the footeye app')
    group = argparser.add_mutually_exclusive_group(required=True)
    group.add_argument('-p',
                       metavar='PROJECT_FILE',
                       help='an existing project file to open')
    group.add_argument('-v',
                       metavar='VIDEO_FILE',
                       help='a video with which to create a new project')
    args = argparser.parse_args()

    if args.p:
        project = loadProject(args.p)
    else:
        project = createProjectFromVideo(args.v)

    processProject(project)


runApp()


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


#cv.destroyAllWindows()
