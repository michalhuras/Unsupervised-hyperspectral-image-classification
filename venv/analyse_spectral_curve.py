#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy import io
import numpy as np
import math

from drafts.tests.test_dataloader import Dataloader as test_dataloader
from dataloader.indian_pines_dataloader import Dataloader as indian_pines_dataloader
from dataloader.samson_dataloader import Dataloader as samson_dataloader
from dataloader.jasper_ridge_dataloader import Dataloader as jasper_ridge_dataloader
from dataloader.salinas_dataloader import Dataloader as salinas_dataloader
from dataloader.salinas_a_dataloader import Dataloader as salinas_a_dataloader
from dataloader.pavia_dataloader import Dataloader as pavia_dataloader


def count_number_of_labels(ideal_spectral_curve):
    return ideal_spectral_curve.shape[0]


def plot(img):
    import matplotlib.pyplot as plt
    plt.plot(img)
    plt.axis('tight')
    plt.show()


def fill_low_matching(matching_clusters, start_row, start_column, labels=[]):
    # TODO delete
    # for 3 labels only
    number_of_labels = 3
    half_fractional_label = 3
    for row in range(half_fractional_label):
        second_row = half_fractional_label + row
        for i in range(number_of_labels):
            matching_clusters[start_row + row][start_column + i] = \
                labels[(row + i) % number_of_labels]
            matching_clusters[start_row + second_row][start_column + number_of_labels - i - 1] = \
                labels[(row + i) % number_of_labels]

    return matching_clusters


def fill_matching_clusters_initial_values(matching_clusters, number_of_labels, labels=[], start_row=0, start_column=0):
    # TODO delete
    if not labels:
        labels = [label for label in range(number_of_labels)]

    if number_of_labels == 3:
        fill_low_matching(matching_clusters, 0, 0, labels)
        return matching_clusters

    row = 0
    for label in labels:
        new_labels = labels.copy()
        new_labels.remove(label)
        if number_of_labels - 1 == 3:
            fill_low_matching(matching_clusters, start_row + row, start_column + 1, new_labels)
        else:
            fill_matching_clusters_initial_values(
                matching_clusters,
                number_of_labels - 1,
                new_labels,
                start_row=row,
                start_column=start_column + 1)

        for i in range(math.factorial(number_of_labels - 1)):
            matching_clusters[start_row + row][start_column] = label
            row += 1

    return matching_clusters

if __name__ == '__main__':
    # Available results files and dataloaders:
    # "./results/IndianPines/data/"     indian_pines_dataloader()
    # "./results/JasperRidge/data/"     jasper_ridge_dataloader()
    # "./results/Pavia/data/"           pavia_dataloader()
    # "./results/Salinas/data/"         salinas_dataloader()
    # "./results/SalinasA/data/"        salinas_a_dataloader()
    # "./results/Samson/data/"          samson_dataloader()
    # "./result/tests/data"           test_dataloader()

    # Searching result files
    # Running function for all result files

    print("START")

    print("* Creating spectral curve loader")
    from dataloader.spectral_curve_dataloader import Dataloader as spectral_curve_dataloader
    dataloader = spectral_curve_dataloader()

    print()
    print("* Loading ideal spectral curve")
    # ideal_file_name = "./results/Samson/data/IDEAL_spectral_curve.txt"
    ideal_file_name = "./results/JasperRidge/data/IDEAL_spectral_curve.txt"
    ideal_spectral_curve = dataloader.get_spectral_curve_from_file(ideal_file_name, verbal=False)
    number_of_labels = count_number_of_labels(ideal_spectral_curve)
    ideal_spectral_curve = ideal_spectral_curve.transpose()
    print("Shape: ", ideal_spectral_curve.shape)
    print("Number of labels: ", number_of_labels)

    print()
    print("* Loading tested spectral curve")
    # file_name = "./results/Samson/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    file_name = "./results/JasperRidge/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    spectral_curve = dataloader.get_spectral_curve_from_file(file_name, verbal=False)
    number_of_labels = count_number_of_labels(spectral_curve)
    spectral_curve = spectral_curve.transpose()
    print("Shape: ", spectral_curve.shape)
    print("Number of labels: ", number_of_labels)

    print()
    print("* Creating array with matching clusters")
    print(type(math.factorial(number_of_labels)))
    print(type(number_of_labels))

    # RODO delete -----
    number_of_labels = 10
    matching_clusters = np.zeros((math.factorial(number_of_labels), number_of_labels + 1))

    matching_clusters = fill_matching_clusters_initial_values(matching_clusters, number_of_labels)
    # matching_clusters = fill_low_matching(matching_clusters, 0, 0)
    print(matching_clusters)
    print("Dlugosć: ", len(matching_clusters))
    # RODO delete ----- end

    # Searching the best result
    # Saving result

    print("END")
