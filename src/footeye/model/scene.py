from enum import Enum


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
