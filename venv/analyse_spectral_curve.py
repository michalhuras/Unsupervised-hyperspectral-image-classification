#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy import io
import numpy as np
import math
import matplotlib.pyplot as plt

import scripts_my.prepare_result_data as prd

from drafts.tests.test_dataloader import Dataloader as test_dataloader
from dataloader.indian_pines_dataloader import Dataloader as indian_pines_dataloader
from dataloader.samson_dataloader import Dataloader as samson_dataloader
from dataloader.jasper_ridge_dataloader import Dataloader as jasper_ridge_dataloader
from dataloader.salinas_dataloader import Dataloader as salinas_dataloader
from dataloader.salinas_a_dataloader import Dataloader as salinas_a_dataloader
from dataloader.pavia_dataloader import Dataloader as pavia_dataloader

from algorithms.pairing_greedy_algorithm import PairingAlgorithm


def compare_with_ground_truth(labeled_image, dataloader, pairs, plot=True):
    print()
    print()
    print("* Compare with ground truth")
    ground_truth = dataloader.get_labels(verbal=False)

    if plot:
        plt.clf()
        plt.subplot(1, 2, 1)
        plt.title('Labeled corrected image')
        plt.imshow(labeled_image)
        plt.subplot(1, 2, 2)
        plt.title('Ground truth')
        plt.imshow(ground_truth)
        plt.show()


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

    file_spectral = "./results/JasperRidge/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    file_labels = "./results/JasperRidge/data/clustering_kmeans_linear_autoencoder_1.txt"
    pairs = prd.pairs_in_spectral_curves(jasper_ridge_dataloader(), file_spectral, PairingAlgorithm())
    labeled_image = prd.get_labeled_image(file_labels, pairs)
    compare_with_ground_truth(labeled_image, jasper_ridge_dataloader(), pairs)

    print("END")
