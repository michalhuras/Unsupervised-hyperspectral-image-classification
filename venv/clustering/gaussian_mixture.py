#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy import linalg
from sklearn import mixture


def clustering(the_image_autoencoded, the_image_shape, number_of_clusters):
    n_components = number_of_clusters
    print()
    print("***   Gaussian Mixture Model Selection   ***")
    print("---------------------------------")
    # https://towardsdatascience.com/the-5-clustering-algorithms-data-scientists-need-to-know-a36d136ef68
    # https://scikit-learn.org/stable/modules/mixture.html
    # https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html
    # #sphx-glr-auto-examples-mixture-plot-gmm-selection-py

    print("Image shape: ", the_image_shape)
    # print("Creating dataframe")
    # df = DataFrame(data=the_image_autoencoded)


    #
    # x = 0
    # y = 0
    # for i in range(the_image_shape[0] * the_image_shape[1]):
    #     clustered_data[y, x] = kmeans.predict([the_image_autoencoded[y * the_image_shape[1] + x]])
    #     x = x + 1
    #     if x == the_image_shape[1]:
    #         x = 0
    #         y = y + 1

    print("Running fit function for Gaussian Mixture Model Selection ")
    # cv_types = ['spherical', 'tied', 'diag', 'full']
    cv_type = 'spherical'
    # TODO sparametryzować

    #  Fit a Gaussian mixture with EM
    clust = mixture.GaussianMixture(n_components=n_components,
                                  covariance_type=cv_type)
    clust.fit(the_image_autoencoded)

    print("Creating list for clustered data")
    clustered_data = np.zeros((the_image_shape[0], the_image_shape[1]))
    print("Clustered data shape:  ", np.shape(clustered_data))

    x = 0
    y = 0
    for i in range(np.shape(clustered_data)[0] * np.shape(clustered_data)[1]):
        clustered_data[y][x] = clust.predict(the_image_autoencoded[i].reshape(1, -1))
        x = x + 1
        if x == the_image_shape[1]:
            x = 0
            y = y + 1

    return clustered_data


def get_name():
    return "gaussian_mixture_model_selection"



