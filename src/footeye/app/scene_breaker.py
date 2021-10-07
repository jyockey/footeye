from footeye.app.model import Scene

import footeye.visionlib.features as features
import cv2 as cv

MIN_SCENE_LENGTH = 20


def break_scenes(project):
    scenes = []
    last_frame = None
    frame_idx = 0
    vid = cv.VideoCapture(project.vidinfo.vidFilePath)
    while vid.isOpened():
        ret, frame = vid.read()
        frame_idx = frame_idx + 1
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        cv.imshow('frame', frame)
        is_break = features.scene_break_check(hsv, last_frame)
        prev_scene_length = 0
        if is_break and scenes:
            prev_scene_length = frame_idx - scenes[-1].frame_start
            if prev_scene_length < MIN_SCENE_LENGTH:
                is_break = False
        if is_break:
            if scenes:
                scenes[-1].frame_count = prev_scene_length
            scenes.append(Scene(frame_idx))
            print("Scene at " + str(frame_idx))
        if cv.waitKey(0 if is_break else 1) == ord('q'):
            break
        last_frame = hsv
    prev_scene_length = frame_idx - scenes[-1].frame_start
    scenes[-1].frame_count = prev_scene_length
    print(scenes)
    vid.release()
    return scenes
