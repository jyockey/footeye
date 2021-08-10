from enum import Enum
from hashlib import blake2b
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

    def save(self):
        filename = VidInfo._hashName(self.vidFilePath) + ".vidinfo"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def _hashName(filePath):
        h = blake2b(digest_size=16)
        h.update(filePath.encode('utf-8'))
        return h.hexdigest()

    @staticmethod
    def forPath(vidFilePath):
        filename = VidInfo._hashName(vidFilePath) + ".vidinfo"
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                return pickle.load(f)
        return VidInfo(vidFilePath)


class FooteyeProject:
    project_name = None
    vidinfo = None
    home_team = None
    away_team = None

    def __init__(self, project_name):
        self.project_name = project_name

    def filename(self):
        return self.project_name + ".fae"

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
