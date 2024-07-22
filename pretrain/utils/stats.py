from PIL import ImageStat
import numpy as np

class Stats(ImageStat.Stat):
    def __add__(self, other):
        return Stats(list(map(np.add, self.h, other.h)))