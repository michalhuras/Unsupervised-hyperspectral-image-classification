#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame
from sklearn.cluster import MeanShift


def clustering(the_image_autoencoded, the_image_shape, number_of_clusters):
    print()
    print("***   Mean-Shift clustering   ***")
    print("---------------------------------")
    # https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html
    print("Image shape: ", the_image_shape)
    # print("Creating dataframe")
    # df = DataFrame(data=the_image_autoencoded)

    print("Running fit function for mean-shift clustering")
    clust = MeanShift(bandwidth=2).fit(the_image_autoencoded)

    print("Creating list for clustered data")
    clustered_data = np.zeros((the_image_shape[0], the_image_shape[1]))
    print("Clustered data shape:  ", np.shape(clustered_data))

    x = 0
    y = 0
    for i in range(the_image_shape[0] * the_image_shape[1]):
        clustered_data[y, x] = clust.labels_[i]
        x = x + 1
        if x == the_image_shape[1]:
            x = 0
            y = y + 1

    # Parameters start
    print("Parameters for this estimation: ", clust.get_params())
    label_min = 1
    label_max = 0
    for i in range(np.shape(clustered_data)[0] * np.shape(clustered_data)[1]):
        if clust.labels_[i] > label_max:
            label_max = clust.labels_[i]
        if clust.labels_[i] < label_min:
            label_min = clust.labels_[i]
    print("Labels from", label_min, " ,to", label_max, ". Number of labels: ", label_max - label_min)
    # Parameters stop

    return clustered_data


def get_name():
    return "clustering_mean_shift"
