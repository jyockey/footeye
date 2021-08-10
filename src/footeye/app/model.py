from enum import Enum
import cv2 as cv
import pickle
import os.path


class VidInfo:
    vidFilePath = None
    frameCount = 0
    fieldColorExtents = None
    scenes = []

    def __init__(self, vidFilePath):
        if not os.path.isfile(vidFilePath):
            raise "No such file: " + vidFilePath

        self.vidFilePath = vidFilePath

        vid = cv.VideoCapture(vidFilePath)
        frameCount = vid.get(cv.CAP_PROP_FRAME_COUNT)
        vid.release()
        self.frameCount = frameCount


class Project:
    project_name = None
    vidinfo = None
    home_team = None
    away_team = None

    def __init__(self, project_name, vidinfo):
        self.project_name = project_name
        self.vidinfo = vidinfo

    def filename(self):
        return self.project_name + ".fey"

    def save(self):
        with open(self.filename(), 'wb') as f:
            pickle.dump(self, f)


class SceneType(Enum):
    LIVE_GAMEPLAY = 1
    REPLAY = 2
    OTHER_INGAME = 3
    OTHER_NONGAME = 4


class ShotType(Enum):
    WIDE = 1
    CLOSE_UP = 2


class Scene:
    sceneType = None
    shotType = None

    def __init__(self, frameStart, frameCount):
        self.frameStart = frameStart
        self.frameCount = frameCount
