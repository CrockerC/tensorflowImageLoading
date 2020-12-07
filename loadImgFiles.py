from typing import List
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import threading


class loadImgFiles:
    def __init__(self, paths: List[str], img_shape=(75, 75, 3)):
        self.__paths = paths
        self.__img_shape = img_shape
        self.__images = [0] * len(self.__paths)

    def getImages(self):
        threads = []

        for i, path in enumerate(self.__paths):
            thread = threading.Thread(target=self.__loadImage, args=(path, i))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return np.array(self.__images)

    def __loadImage(self, path, ind):
        img = load_img(path, target_size=self.__img_shape)  # this is a PIL image
        x = np.array(img)  # Numpy array with shape shape
        x = x / 255

        self.__images[ind] = x
