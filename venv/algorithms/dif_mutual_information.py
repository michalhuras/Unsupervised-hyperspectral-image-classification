#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_regression, mutual_info_regression

'''
    Counting difference between spectral curves
    Method: Mutual information
'''


def count_difference(array_1, array_2, show_img=False):
    if show_img:
        plt.clf()
        plt.title("Porównywanie krzywych spektralnych, obliczanie różnicy")
        plt.plot(array_1, label="Array 1")
        plt.plot(array_2, label="Array 2")
        plt.legend()
        plt.axis('tight')
        plt.show()

    X = np.arange(len(array_1))

    mi1 = mutual_info_regression(X.reshape(-1, 1), array_1)
    mi2 = mutual_info_regression(X.reshape(-1, 1), array_2)

    result = np.square(mi1 - mi2)

    # mi1 /= np.max(mi1)
    return result * 1000


'''
    np.random.seed(0)
    X = np.random.rand(1000, 3)
    y = X[:, 0] + np.sin(6 * np.pi * X[:, 1]) + 0.1 * np.random.randn(1000)
    
    f_test, _ = f_regression(X, y)
    f_test /= np.max(f_test)
    
    
    
    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.scatter(X[:, i], y, edgecolor='black', s=20)
        plt.xlabel("$x_{}$".format(i + 1), fontsize=14)
        if i == 0:
            plt.ylabel("$y$", fontsize=14)
        plt.title("F-test={:.2f}, MI={:.2f}".format(f_test[i], mi[i]),
                  fontsize=16)
    plt.show()
'''