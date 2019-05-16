import matplotlib.pyplot as plt
from PIL import Image
import numpy as np


class Digit:
    """
    Represents scanned image of the handwritten digit
    """
    # Path to scanned handwritten digits
    DIGITS_PATH = "digits/"
    LAST_ID = 0

    def __init__(self, file_name):
        """
        Reads the image from file_name and stores it in the array self.image
        :param file_name: name of the file
        """
        im = Image.open(self.DIGITS_PATH + file_name)
        self.image = np.asarray(im)
        self.id = self.LAST_ID + 1
        self.LAST_ID = self.LAST_ID + 1

    def crop(self):
        """
        Deletes the whitespaces around handwritten digit
        """
        nr = self.image.shape[0]
        nc = self.image.shape[1]
        t = 252
        top = 0
        while top < nr and ((np.sum(self.image[top, 0:nc]) / nc > t) or
                            (np.sum(self.image[top + 1, 0:nc]) / nc <= t < np.sum(self.image[top + 1, 0:nc]) / nc)):
            top = top + 1

        bottom = nr - 1

        while (bottom > top) and ((np.sum(self.image[bottom, 0:nc]) / nc > t) or
                                  (np.sum(self.image[bottom, 0:nc]) / nc <= t < np.sum(
                                      self.image[bottom - 1, 0:nc]) / nc)):
            bottom = bottom - 1

        left = 0
        while (left < nc) and ((np.sum(self.image[0:nr, left]) / nr > t) or
                               (np.sum(self.image[0:nr, left]) / nr <= t < np.sum(self.image[0:nr, left + 1]) / nr)):
            left = left + 1

        right = nc - 1
        while (right > left) and ((np.sum(self.image[0:nr, right]) / nr > t) or
                                  (np.sum(self.image[0:nr, right]) / nr <= t < np.sum(
                                      self.image[0:nr, right - 1]) / nr)):
            right = right - 1

        self.image = self.image[top:bottom, left:right]

    def binarize(self, treshold=0.2):
        """
        Binarizes the image based on a treshold
        :param treshold: Binarization treshold
        :return:
        """
        img = np.double(self.image)
        img = 255 - img
        img[img < treshold * np.max(img)] = 0
        img[img >= treshold * np.max(img)] = 255
        img = 255 - img
        img = np.uint8(img)
        self.image = img

    def features(self, custom_func=None, crop_image=False, binarize_image=False, num_of_features=2):
        """
        :param custom_func: user can provide custom function for calculating features. Useful for testing different
        types of features while developing classifier. Custom function should accept numpy matrix as parameter
        :param crop_image: True if image should be cropped (remove whitespaces around handwritten digit) with default
        parameters before calculating features, False otherwise
        :param binarize_image: True if image should be binarized with default parameters before calculating features,
        False otherwise
        :param num_of_features: number of features, used for plotting
        :return: array of features
        """
        features = np.zeros((num_of_features, 1))
        if binarize_image:
            self.binarize()
        if crop_image:
            self.crop()
        if custom_func:
            features = custom_func(self.image)
        else:
            nr = self.image.shape[0]
            nc = self.image.shape[1]
            features[0, 0] = np.mean(self.image[0:round((nr - 1) / 2), :])
            features[1, 0] = np.mean(self.image[round((nr - 1) / 2):-1, 0:round((nc - 1) / 4)])
        return features

    def show(self):
        plt.imshow(self.image, cmap='Greys_r')
        plt.show()

    def save(self, file_path):
        result = Image.fromarray(self.image)
        result.save(file_path)
