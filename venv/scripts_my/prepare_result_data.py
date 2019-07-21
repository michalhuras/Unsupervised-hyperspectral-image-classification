#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy import io
import numpy as np
import math
import matplotlib.pyplot as plt


def count_number_of_labels(ideal_spectral_curve):
    return ideal_spectral_curve.shape[0]


def plot(img):
    import matplotlib.pyplot as plt
    plt.plot(img)
    plt.axis('tight')
    plt.show()


def get_labeled_image(labeled_image_path, pairs, plot=False):
    print("** Get labeled image")
    print("* Creating result dataloader")
    from dataloader.result_dataloader import Dataloader as result_dataloader
    result_dataloader = result_dataloader()
    labeled_image_f = result_dataloader.get_image_labels_from_file(labeled_image_path)

    transpose_pairs = np.zeros(np.shape(pairs))
    for i in pairs:
        transpose_pairs[int(pairs[i])] = i

    corrected_labeled_image = np.zeros(labeled_image_f.shape)
    for y in range(labeled_image_f.shape[0]):
        for x in range(labeled_image_f.shape[1]):
            corrected_labeled_image[y][x] = int(transpose_pairs[int(labeled_image_f[y][x])])

    if plot:
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.imshow(labeled_image_f)
        plt.title('Labeled image')
        plt.subplot(1, 2, 2)
        plt.imshow(corrected_labeled_image)
        plt.title('Labeled corrected image')
        plt.show()

    return corrected_labeled_image


def pairs_in_spectral_curves(ideal_dataloader, file_name, pairing_obj):
    print("* Creating spectral curve loader")
    from dataloader.spectral_curve_dataloader import Dataloader as spectral_curve_dataloader
    dataloader = spectral_curve_dataloader()

    print()
    print("* Loading ideal spectral curve")
    # ideal_file_name = "./results/Samson/data/IDEAL_spectral_curve.txt"
    # ideal_file_name = "./results/JasperRidge/data/IDEAL_spectral_curve.txt"
    ideal_file_name = ideal_dataloader.get_results_directory(verbal=False) + "data/IDEAL_spectral_curve.txt"
    ideal_spectral_curve = dataloader.get_spectral_curve_from_file(ideal_file_name, verbal=False)
    number_of_labels = count_number_of_labels(ideal_spectral_curve)
    ideal_spectral_curve = ideal_spectral_curve.transpose()
    print("Shape: ", ideal_spectral_curve.shape)
    print("Number of labels: ", number_of_labels)

    print()
    print("* Loading tested spectral curve")
    # file_name = "./results/Samson/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    # file_name = "./results/JasperRidge/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    spectral_curve = dataloader.get_spectral_curve_from_file(file_name, verbal=False)
    number_of_labels = count_number_of_labels(spectral_curve)
    spectral_curve = spectral_curve.transpose()
    print("Shape: ", spectral_curve.shape)
    print("Number of labels: ", number_of_labels)

    print()
    print("* Pairing labels")
    f_pairs, difference_value =\
        pairing_obj.math_in_pairs(number_of_labels, ideal_spectral_curve, spectral_curve, prefix="\t", verbal=False)
    f_pairs = [int(element) for element in f_pairs]
    print("Pairs: ", f_pairs)
    print("Difference value: ", difference_value)

    print()
    print("Saving paired labels")
    results_dir = ideal_dataloader.get_results_directory(verbal=False)
    result_name = file_name.split("/")[-1]
    result_name = result_name.split(".")[0]
    result_name += "_pairs.txt"
    result_data_path = results_dir + "data/" + result_name
    print("Result data path:", result_data_path)
    np.savetxt(result_data_path, f_pairs, delimiter=" ", newline="\n", header=result_name, fmt="%s")

    return f_pairs
