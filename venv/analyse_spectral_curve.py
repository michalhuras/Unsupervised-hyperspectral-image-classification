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


def image_to_list(image):
    result_list = np.zeros(image.shape[0] * image.shape[1])
    i = 0
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            result_list[i] = image[y][x]
            i += 1
    return result_list


def create_confusion_matrix(labeled_image, ground_truth):
    labeled_image_list = image_to_list(labeled_image)
    ground_truth_list = image_to_list(ground_truth)
    print()
    print("CONFUSION MATRIX")
    from sklearn.metrics import multilabel_confusion_matrix
    confusion_matrix = multilabel_confusion_matrix(ground_truth_list, labeled_image_list)
    print(confusion_matrix)

    print()
    print("PRECISION")
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score, f1_score
    print("Precision: ", precision_score(ground_truth_list, labeled_image_list, average=None))
    print("recall: ", recall_score(ground_truth_list, labeled_image_list, average=None))
    print("F1: ", f1_score(ground_truth_list, labeled_image_list, average=None))
    '''
    print("Macro - counted for each label, and found unweighted mean")
    print(precision_score(ground_truth_list, labeled_image_list, average='macro'))
    print("Micro - counted globally")
    print(precision_score(ground_truth_list, labeled_image_list, average='micro'))
    print("Weighted")
    print(precision_score(ground_truth_list, labeled_image_list, average='weighted'))
    print(precision_score(ground_truth_list, labeled_image_list, average=None))
    '''


def compare_with_ground_truth(labeled_image, dataloader, pairs, plot=False):
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

    create_confusion_matrix(labeled_image, ground_truth)


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
