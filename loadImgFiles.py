from typing import List
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import threading


class loadImgFiles:
    def __init__(self, paths: List[str], img_shape=(224, 224, 3)):
        self.__paths = paths
        self.__img_shape = img_shape
        self.__sem = threading.Semaphore()
        self.__images = []

    def getImages(self):
        threads = []

        for path in self.__paths:
            thread = threading.Thread(target=self.__loadImage, args=(path,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

        return np.array(self.__images)

    def __loadImage(self, path):
        img = load_img(path, target_size=self.__img_shape)  # this is a PIL image
        x = np.array(img)  # Numpy array with shape shape
        x = x / 255

        self.__images.append(x)

