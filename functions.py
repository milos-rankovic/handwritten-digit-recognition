import numpy as np
from digit import Digit
import matplotlib.pyplot as plt
import math

# Number of features
NUM_OF_FEAT = 2


def features(img):
    """
    :param img: Image for which features will be returned. Image should be passed as python 2d array
    :return: NumPy array of features
    """
    nr = img.shape[0]
    nc = img.shape[1]
    feat1 = np.mean(np.mean(img[1:round(nr/2), round(nc*3/4):])) - np.mean(np.mean(img[round(nr / 2):, 1: round(nc / 4)]))
    feat2 = np.mean(np.mean(img[round(nr*(2/7)):round(nr*(5/7)), round(nc*(3/7)):round(nc*(4/7))]))
    return np.array([[feat1], [feat2]])


def calculate_features(N):
    """
    :param N: Number of images for each digit
    :return: Features for each image as NumPy array
    """
    feat_6 = np.zeros((NUM_OF_FEAT, N))
    feat_8 = np.zeros((NUM_OF_FEAT, N))
    feat_9 = np.zeros((NUM_OF_FEAT, N))
    for i in range(0, N):
        file_name = "baza6" + format(i + 1, "03d") + ".bmp"
        digit = Digit(file_name)
        digit.binarize()
        digit.crop()
        feat_6[:, i] = np.transpose(digit.features(custom_func=features, num_of_features=NUM_OF_FEAT))

        file_name = "baza8" + format(i + 1, "03d") + ".bmp"
        digit = Digit(file_name)
        digit.binarize()
        digit.crop()
        feat_8[:, i] = np.transpose(digit.features(custom_func=features, num_of_features=NUM_OF_FEAT))

        file_name = "baza9" + format(i + 1, "03d") + ".bmp"
        digit = Digit(file_name)
        digit.binarize()
        digit.crop()
        feat_9[:, i] = np.transpose(digit.features(custom_func=features, num_of_features=NUM_OF_FEAT))
    return feat_6, feat_8, feat_9


def training(training_set_6, training_set_8, training_set_9):
    """
    Based on a training sets of patterns (features) estimates statistics, such as expected value vector and covariance matrix
    :param training_set_6: training set for digit 6
    :param training_set_8: training set for digit 8
    :param training_set_9: training set for digit 9
    :return: expected values and covariance matrices for each pattern (random vectors - features), for class 1, 2 and 3
    """
    # Statistics estimation, assume Normal distribution
    m11 = np.mean(training_set_6[0, :])
    m21 = np.mean(training_set_6[1, :])
    M1 = [[m11], [m21]]
    S1 = np.cov(training_set_6)

    m12 = np.mean(training_set_8[0, :])
    m22 = np.mean(training_set_8[1, :])
    M2 = [[m12], [m22]]
    S2 = np.cov(training_set_8)

    m13 = np.mean(training_set_9[0, :])
    m23 = np.mean(training_set_9[1, :])
    M3 = [[m13], [m23]]
    S3 = np.cov(training_set_9)
    return M1, S1, M2, S2, M3, S3


def show_scatter(training_set_6, training_set_8, training_set_9):
    if NUM_OF_FEAT == 2:
        plt.figure(0)
        plt.scatter(training_set_6[0, :], training_set_6[1, :], color="blue", marker="*", label="6")
        plt.scatter(training_set_8[0, :], training_set_8[1, :], color="green", marker="x", label="8")
        plt.scatter(training_set_9[0, :], training_set_9[1, :], color="red", marker="o", label="9")
        plt.legend(loc="lower left")
        plt.show()


def bayes_test(test_set_6, test_set_8, test_set_9, M1, S1, M2, S2, M3, S3):
    """
    :param test_set_6: test set for digit 6
    :param test_set_8: test set for digit 8
    :param test_set_9: test set for digit 9
    :param M1: expected value of the first pattern (random vector for the class 1 - digit 6)
    :param S1: covariance matrix of the first pattern (random vector for the class 1 - digit 6)
    :param M2: expected value of the second pattern (random vector for the class 2 - digit 8)
    :param S2: covariance matrix of the second pattern (random vector for the class 3 - digit 8)
    :param M3: expected value of the third pattern (random vector for the class 3 - digit 9)
    :param S3: covariance matrix of the third pattern (random vector for the class 3 - digit 9)
    :return: confusion matrix, which represents how good is the test
    """
    conf_mat = np.zeros((3, 3))

    for class_number in range(0, 3):
        if class_number == 0:
            test_set = test_set_6
        elif class_number == 1:
            test_set = test_set_8
        else:
            test_set = test_set_9

        for i in range(0, test_set.shape[1]):
            f1 = math.exp(-0.5 * np.transpose((test_set[:, i].reshape(2, 1) - M1)) @ np.linalg.inv(S1) @ (
                        test_set[:, i].reshape(2, 1) - M1)) / (2 * np.pi * np.sqrt(np.linalg.det(S1)))
            f2 = math.exp(-0.5 * np.transpose((test_set[:, i].reshape(2, 1) - M2)) @ np.linalg.inv(S2) @ (
                        test_set[:, i].reshape(2, 1) - M2)) / (2 * np.pi * np.sqrt(np.linalg.det(S2)))
            f3 = math.exp(-0.5 * np.transpose((test_set[:, i].reshape(2, 1) - M3)) @ np.linalg.inv(S3) @ (
                        test_set[:, i].reshape(2, 1) - M3)) / (2 * np.pi * np.sqrt(np.linalg.det(S3)))

            if f1 > f2 and f1 > f3:
                conf_mat[class_number, 0] += 1
            elif f2 > f1 and f2 > f3:
                conf_mat[class_number, 1] += 1
            elif f3 > f1 and f3 > f2:
                conf_mat[class_number, 2] += 1
            else:
                print("Classification failed")
    return conf_mat
