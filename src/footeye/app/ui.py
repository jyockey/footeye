import cv2 as cv


def play(project, trans_function=None, scene=None, loop=True):
    vid = cv.VideoCapture(project.vidinfo.vidFilePath)
    frame_start = scene.frame_start if scene else 1
    frame_count = scene.frame_count if scene else vid.get(cv.CAP_PROP_FRAME_COUNT)

    while True:
        idx = 0
        vid.set(cv.CAP_PROP_POS_FRAMES, frame_start)
        print(frame_start)
        print(frame_count)
        while (idx < frame_count):
            ret, frame = vid.read()
            # if frame is read correctly ret is True
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            frame = trans_function(frame) if trans_function else frame
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                break
            idx = idx + 1
        if not loop:
            break
    vid.release()


def play_scene(project, scene,  loop=True):
    vid = cv.VideoCapture(project.vidinfo.vidFilePath)

    while True:
        idx = 1
        vid.set(cv.CAP_PROP_POS_FRAMES, scene.frame_start)
        while idx < scene.frame_count:
            ret, frame = vid.read()
            if frame is None:
                print("Got no frame for idx " + str(scene.frame_start + idx))
                return
            cv.imshow('frame', frame)
            if cv.waitKey(1) == ord('q'):
                return
            idx = idx + 1
        if not loop:
            break
