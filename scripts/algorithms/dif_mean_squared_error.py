#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


def count_difference(array_1, array_2, show_img=False):
    # Mean squared error
    # https://www.geeksforgeeks.org/python-mean-squared-error/
    if show_img:
        plt.clf()
        plt.title("Porównywanie krzywych spektralnych, obliczanie różnicy")
        plt.plot(array_1, label="Array 1")
        plt.plot(array_2, label="Array 2")
        plt.legend()
        plt.axis('tight')
        plt.show()

    result = (np.square(array_1 - array_2)).mean(axis=None)

    # from sklearn.feature_selection import mutual_info_classif
    # result = mutual_info_classif(array_1, array_2)

    # print("Value: \t\t", mse)
    return result * 1000

    # Old method:
    # return np.square(np.subtract(array_1, array_2)) #.mean()