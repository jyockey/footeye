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
    scenes = []

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
    scene_type = None
    shot_type = None
    frame_start = 0
    frame_count = 0
    frames = []

    def __init__(self, frame_start):
        self.frame_start = frame_start

    def __str__(self):
        return "SCENE {0} [{1}]".format(self.frame_start, self.frame_count)

    __repr__ = __str__


class FrameInfo:
    frame_idx = None
    timestamp = None
    raw_features = []
    player_entities = []
    ball_entity = None
    player_size = None
    camera_angle = None

    def __init__(self, frame_idx):
        self.frame_idx = frame_idx
