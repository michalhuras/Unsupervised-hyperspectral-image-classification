#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def clustering(the_image_autoencoded, the_image_shape, number_of_clusters):
    print()
    print("***   K - means clustering   ***")
    print("---------------------------------")
    # https://www.datacamp.com/community/tutorials/k-means-clustering-python
    # https: // datatofish.com / k - means - clustering - python /

    print("Image shape: ", the_image_shape)
    print("Creating dataframe from k-clustering")
    df = DataFrame(data=the_image_autoencoded)

    print("KMeans clustering")
    kmeans = KMeans(n_clusters=number_of_clusters).fit(df)

    print("Creating list for clustered data")
    clustered_data = np.zeros((the_image_shape[0], the_image_shape[1]))
    print("Clustered data shape:  ", np.shape(clustered_data))

    x = 0
    y = 0
    for i in range(the_image_shape[0] * the_image_shape[1]):
        clustered_data[x][y] = kmeans.predict([the_image_autoencoded[y * the_image_shape[0] + x]])
        x = x + 1
        if x == the_image_shape[1]:
            x = 0
            y = y + 1

    return clustered_data


def get_name():
    return "clustering_kmeans"
