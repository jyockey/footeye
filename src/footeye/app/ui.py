import cv2 as cv


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
