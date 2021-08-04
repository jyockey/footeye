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
