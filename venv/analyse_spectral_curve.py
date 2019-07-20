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

from algorithms.pairing_greedy_algorithm import PairingAlgorithm


def count_number_of_labels(ideal_spectral_curve):
    return ideal_spectral_curve.shape[0]


def plot(img):
    import matplotlib.pyplot as plt
    plt.plot(img)
    plt.axis('tight')
    plt.show()


def pairs_in_spectral_curves(ideal_dataloader, file_name):
    print("* Creating spectral curve loader")
    from dataloader.spectral_curve_dataloader import Dataloader as spectral_curve_dataloader
    dataloader = spectral_curve_dataloader()

    print()
    print("* Loading ideal spectral curve")
    # ideal_file_name = "./results/Samson/data/IDEAL_spectral_curve.txt"
    # ideal_file_name = "./results/JasperRidge/data/IDEAL_spectral_curve.txt"
    ideal_file_name = ideal_dataloader.get_spectral_curve_directory(verbal=False)
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
    pairing_obj = PairingAlgorithm()
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

def compare_with_ground_truth(labeled_image_path, dataloader, pairs):
    print()
    print()
    print("* Compare with ground truth")

    print("* Creating result dataloader")
    from dataloader.result_dataloader import Dataloader as result_dataloader
    dataloader = result_dataloader()


if __name__ == '__main__':
    print("START")

    # Available results files and dataloaders:
    # "./results/IndianPines/data/"     indian_pines_dataloader()
    # "./results/JasperRidge/data/"     jasper_ridge_dataloader()
    # "./results/Pavia/data/"           pavia_dataloader()
    # "./results/Salinas/data/"         salinas_dataloader()
    # "./results/SalinasA/data/"        salinas_a_dataloader()
    # "./results/Samson/data/"          samson_dataloader()
    # "./result/tests/data"           test_dataloader()

    # # Searching result files
    # # Running function for all result files

    file_name = "./results/JasperRidge/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    pairs = pairs_in_spectral_curves(jasper_ridge_dataloader(), file_name)
    compare_with_ground_truth(file_name, jasper_ridge_dataloader(), pairs)

    print("END")
