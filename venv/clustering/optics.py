#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from pandas import DataFrame
from sklearn.cluster import OPTICS, cluster_optics_dbscan

# WIP - zawsze zwraca jedną wartość

def clustering(the_image_autoencoded, the_image_shape, number_of_clusters):
    print()
    print("***   OPTICS clustering   ***")
    print("---------------------------------")
    # https://scikit-learn.org/stable/modules/clustering.html
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html
    # #sphx-glr-auto-examples-cluster-plot-optics-py
    # https://scikit-learn.org/stable/modules/clustering.html#optics

    print("Image shape: ", the_image_shape)

    print("OPTICS clustering")
    clust = OPTICS(min_samples=10, xi=.0005, min_cluster_size=.005)

    print("Running fit function for OPTICS clustering")
    clust.fit(the_image_autoencoded)

    labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=0.5)

    labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=2)

    labels_300 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=3)

    print("---------------------------")
    reachability = clust.reachability_[clust.ordering_]
    print("Reachability: ", reachability)
    print("---------------------------")

    print("Creating list for clustered data")
    clustered_data = np.zeros((the_image_shape[0], the_image_shape[1]))
    print("Clustered data shape:  ", np.shape(clustered_data))

    x = 0
    y = 0
    for i in range(the_image_shape[0] * the_image_shape[1]):
        clustered_data[y, x] = labels_050[y * the_image_shape[1] + x]
        x = x + 1
        if x == the_image_shape[1]:
            x = 0
            y = y + 1

    return clustered_data


def get_name():
    return "clustering_optics"
