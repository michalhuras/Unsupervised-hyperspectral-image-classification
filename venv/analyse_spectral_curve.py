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


def create_confusion_matrix(labeled_image, ground_truth, verbal=False):
    labeled_image_list = image_to_list(labeled_image)
    ground_truth_list = image_to_list(ground_truth)
    if verbal:
        print()
        print("CONFUSION MATRIX")
    from sklearn.metrics import multilabel_confusion_matrix
    confusion_matrix = multilabel_confusion_matrix(ground_truth_list, labeled_image_list)
    if verbal:
        print(confusion_matrix)
    return confusion_matrix


def get_precision(labeled_image, ground_truth, verbal=False):
    labeled_image_list = image_to_list(labeled_image)
    ground_truth_list = image_to_list(ground_truth)
    if verbal:
        print()
        print("PRECISION")
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score, f1_score
    report = "PRECISION"
    report += "\nPrecision: \t\t\t" + str(precision_score(ground_truth_list, labeled_image_list, average=None))
    report += "\nRecall: \t\t\t" + str(recall_score(ground_truth_list, labeled_image_list, average=None))
    report += "\nF1: \t\t\t" + str(f1_score(ground_truth_list, labeled_image_list, average=None))
    '''
    print("Macro - counted for each label, and found unweighted mean")
    print(precision_score(ground_truth_list, labeled_image_list, average='macro'))
    print("Micro - counted globally")
    print(precision_score(ground_truth_list, labeled_image_list, average='micro'))
    print("Weighted")
    print(precision_score(ground_truth_list, labeled_image_list, average='weighted'))
    print(precision_score(ground_truth_list, labeled_image_list, average=None))
    '''
    if verbal:
        print(report)
    return report


def compare_with_ground_truth(labeled_image, dataloader, pairs, plot=False):
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

    report = "Confusion matrix: \n" + str(create_confusion_matrix(labeled_image, ground_truth))
    report += "\n\n\n" + get_precision(labeled_image, ground_truth)
    print("REPORT \n")
    print(report)
    return report


def save_report(report, file_labels):
    print()
    print("* Save report")

    file_labels_split = file_labels.split("/")
    report_name = "report_" + file_labels_split[-1]
    report_path = ""
    for i in range(len(file_labels_split) - 1):
        report_path += file_labels_split[i] + "/"

    print("Report file name: \t\t", report_name)
    print("Report file path: \t\t", report_path)

    with open(report_path + report_name, "w") as text_file:
        print(report, file=text_file)


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
    '''
    import os

    print("Searching for spectral curve and labeled files")
    result_directories_with_dataloaders = {
        "./results/IndianPines/data/": indian_pines_dataloader(),
        "./results/JasperRidge/data/": jasper_ridge_dataloader(),
        "./results/Pavia/data/": pavia_dataloader(),
        "./results/Salinas/data/": salinas_dataloader(),
        "./results/SalinasA/data/": salinas_a_dataloader(),
        "./results/Samson/data/": samson_dataloader(),
        # "./result/tests/data":test_dataloader(),
    }


    # name: directory
    for path in result_directories_with_dataloaders:
        print()
        print()
        names_and_directories = {}
        print("\tPath: ", path)
        print("\tDataloader name: ", result_directories_with_dataloaders[path].get_name(False))

        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if '.txt' in file and "spectral_curve" not in file:
                    names_and_directories[file] = os.path.join(r, file)

        print(names_and_directories)

        for file_name in names_and_directories:
            labels_image_path = names_and_directories[file_name]
            dataloader_for_this = result_directories_with_dataloaders[path]
            labels_path = dataloader_for_this.get_results_directory(verbal=False) + "spectral_curve_" + file_name

            print()
            print("\t File name:  ", file_name)
            print("\t Labels image path: ", labels_image_path)
            print("\t Spectral curve path: ", labels_path)
            print("\t Dataloader name: ", dataloader_for_this.get_name(verbal=False))


            create_spectral_curve_from_dataloader_plus(
                result_directories_with_dataloaders[path],
                image_labels,
                output_name=file_name,
                show_img=False)

           
    print()
    print()
    print()
    print()
    print()
     '''

    # TU się zaczyna funkcja dla jednej wartości !!!!!
    # in dataloader
    # int spectral curve path
    # int image labeled path

    file_spectral = "./results/JasperRidge/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    file_labels = "./results/JasperRidge/data/clustering_kmeans_linear_autoencoder_1.txt"
    print("File spectral curve: \t\t", file_spectral)
    print("File labels: \t\t\t", file_labels)
    pairs = prd.pairs_in_spectral_curves(jasper_ridge_dataloader(), file_spectral, PairingAlgorithm())
    labeled_image = prd.get_labeled_image(file_labels, pairs)
    report = compare_with_ground_truth(labeled_image, jasper_ridge_dataloader(), pairs)
    save_report(report, file_labels)



    print("END")
