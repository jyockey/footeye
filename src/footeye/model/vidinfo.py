from hashlib import blake2b
import pickle
import os.path


class VidInfo:
    vidFilePath = None
    frameCount = 0
    fieldColorExtents = None
    scenes = []

    def __init__(self, vidFilePath):
        self.vidFilePath = vidFilePath

    def save(self):
        filename = VidInfo._hashName(self.vidFilePath) + ".vidinfo"
        with open(filename, 'w') as f:
            pickle.dump(self, f)

    @staticmethod
    def _hashName(filePath):
        h = blake2b(digest_size=16)
        h.update(filePath)
        return h.hexdigest()

    @staticmethod
    def forPath(vidFilePath):
        filename = VidInfo._hashName(vidFilePath) + ".vidinfo"
        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                return pickle.load(f)
        return VidInfo(vidFilePath)
