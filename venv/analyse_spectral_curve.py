#!/usr/bin/env python
# -*- coding: utf-8 -*-

from scipy import io
import os
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

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score, f1_score

def create_result_img_path(base_path):
    base_path_split = base_path.split("/")
    save_name = "comparison_" + base_path_split[-1]
    if save_name.endswith('.txt'):
        save_name = save_name[:-4]
    save_name += ".png"
    save_path = ""
    for i in range(len(base_path_split) - 2):
        save_path += base_path_split[i] + "/"

    save_path = save_path + "comparison/" + save_name

    return save_path


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


def compare_with_ground_truth(
        labeled_image, dataloader, path_to_file, plot=False, verbal=False, save_img=False, result_img_path=""):
    if verbal:
        print()
        print("* Compare with ground truth")
    ground_truth = dataloader.get_labels(verbal=False)

    if plot or save_img:
        # plt.clf()
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(labeled_image)
        axs[0].set_title('Labeled corrected image')
        axs[1].set_title('Ground truth')
        axs[1].imshow(ground_truth)
        fig.suptitle(path_to_file.split("/")[-1], fontsize=16)
        if save_img:
            plt.savefig(result_img_path, bbox_inches='tight')
            print("Saving comparison to file: ", result_img_path)
        # plt.draw()
        elif plot:
            plt.show()

    report = "Confusion matrix: \n" + str(create_confusion_matrix(labeled_image, ground_truth))
    report += "\n\n\n" + get_precision(labeled_image, ground_truth)
    if verbal:
        print("REPORT \n")
        print(report)
    return report


def save_report(report, file_labels, verbal=True):
    if verbal:
        print("* Save report")

    file_labels_split = file_labels.split("/")
    report_name = "report_" + file_labels_split[-1]
    report_path = ""
    for i in range(len(file_labels_split) - 2):
        report_path += file_labels_split[i] + "/"

    save_path = report_path + "comparison/" + report_name
    if verbal:
        # print("Report file name: \t\t", report_name)
        # print("Report file path: \t\t", report_path)
        print("Save path: \t\t", save_path)

    with open(save_path, "w") as text_file:
        print(report, file=text_file)


def single_analyse(dataloader_local, spectral_curve_path, labeled_image_path):
    print("File spectral curve: \t\t", spectral_curve_path)
    print("File labels: \t\t\t", labeled_image_path)
    pairs = prd.pairs_in_spectral_curves(dataloader_local, spectral_curve_path, PairingAlgorithm(), verbal=True)
    labeled_image = prd.get_labeled_image(labeled_image_path, pairs)
    comparison_img_path = create_result_img_path(labeled_image_path)
    report =\
        compare_with_ground_truth(labeled_image, dataloader_local, labeled_image_path,
                                  result_img_path=comparison_img_path, save_img=True)
    save_report(report, labeled_image_path)


def single_analyse_beta(dataloader_local, spectral_curve_path, labeled_image_path):
    print("File spectral curve: \t\t", spectral_curve_path)
    print("File labels: \t\t\t", labeled_image_path)
    properties = []
    pairs = prd.pairs_in_spectral_curves(dataloader_local, spectral_curve_path, PairingAlgorithm(), verbal=True)
    labeled_image = prd.get_labeled_image(labeled_image_path, pairs)

    labeled_image_list = image_to_list(labeled_image)
    ground_truth_list = image_to_list(dataloader_local.get_labels(verbal=False))
    properties.append(str(precision_score(ground_truth_list, labeled_image_list, average='micro')))
    properties.append(str(precision_score(ground_truth_list, labeled_image_list, average='macro')))
    properties.append(str(precision_score(ground_truth_list, labeled_image_list, average='weighted')))
    # properties.append(str(precision_score(ground_truth_list, labeled_image_list, average='samples')))
    properties.append(str(recall_score(ground_truth_list, labeled_image_list, average='micro')))
    properties.append(str(recall_score(ground_truth_list, labeled_image_list, average='macro')))
    properties.append(str(recall_score(ground_truth_list, labeled_image_list, average='weighted')))
    # properties.append(str(recall_score(ground_truth_list, labeled_image_list, average='samples')))
    properties.append(str(f1_score(ground_truth_list, labeled_image_list, average='micro')))
    properties.append(str(f1_score(ground_truth_list, labeled_image_list, average='macro')))
    properties.append(str(f1_score(ground_truth_list, labeled_image_list, average='weighted')))
    # properties.append(str(f1_score(ground_truth_list, labeled_image_list, average='samples')))
    return properties


def analyse_all_data_separately():
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
                    if '.txt' in file and "spectral_curve" not in file and "report_" not in file:
                        names_and_directories[file] = os.path.join(r, file)

            print(names_and_directories)

            for file_name in names_and_directories:
                labels_image_path = names_and_directories[file_name]
                dataloader_for_this = result_directories_with_dataloaders[path]
                spectral_curve_path = \
                    dataloader_for_this.get_results_directory(verbal=False) + "data/spectral_curve_" + file_name

                if os.path.exists(spectral_curve_path):
                    print()
                    print("\t File name:  ", file_name)
                    print("\t Labels image path: ", labels_image_path)
                    print("\t Spectral curve path: ", spectral_curve_path)
                    print("\t Dataloader name: ", dataloader_for_this.get_name(verbal=False))

                    single_analyse(dataloader_for_this, spectral_curve_path, labels_image_path)
                else:
                    print()
                    print("FILE DOES NOT EXIST")
                    print("File: ", spectral_curve_path)


def analyse_all_data_together():
    import csv
    print("Analyze all data together")
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
        print("Create comparison file")
        comparison_file_name =\
            result_directories_with_dataloaders[path].get_results_directory() + \
            "comparison_" + \
            result_directories_with_dataloaders[path].get_name() + \
            ".csv"
        print("Comparison file name: ", comparison_file_name)

        with open(comparison_file_name, 'w', newline='') as csvfile:
            filewriter = csv.writer(csvfile, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            filewriter.writerow(['Name',
                                 "Precision(micro)", "Precision(macro)", "Precision(weighted)",
                                 "Recall(micro)", "Recall(macro)", "Recall(weighted)",
                                 "F1(micro)", "F1(macro)", "F1(weighted)"])

            # r=root, d=directories, f = files
            for r, d, f in os.walk(path):
                for file in f:
                    if '.txt' in file and "spectral_curve" not in file and "report_" not in file:
                        names_and_directories[file] = os.path.join(r, file)

            print(names_and_directories)

            for file_name in names_and_directories:
                labels_image_path = names_and_directories[file_name]
                dataloader_for_this = result_directories_with_dataloaders[path]
                spectral_curve_path = \
                    dataloader_for_this.get_results_directory(verbal=False) + "data/spectral_curve_" + file_name

                if os.path.exists(spectral_curve_path):
                    print()
                    print("\t File name:  ", file_name)
                    print("\t Labels image path: ", labels_image_path)
                    print("\t Spectral curve path: ", spectral_curve_path)
                    print("\t Dataloader name: ", dataloader_for_this.get_name(verbal=False))

                    report_list = single_analyse_beta(dataloader_for_this, spectral_curve_path, labels_image_path)
                    report_list = [file_name] + report_list
                    filewriter.writerow(report_list)
                else:
                    print()
                    print("FILE DOES NOT EXIST")
                    print("File: ", spectral_curve_path)


if __name__ == '__main__':
    to_file = False
    # Przekierowanie wyjścia do pliku
    if to_file:
        import sys
        orig_stdout = sys.stdout
        output_file = open('results/analyse_spectral_curve_output.txt', 'w')
        sys.stdout = output_file

    print("START")

    # analyse_all_data_separately()
    analyse_all_data_together()

    '''
    # # Available results files and dataloaders:
    # "./results/IndianPines/data/"     indian_pines_dataloader()
    # "./results/JasperRidge/data/"     jasper_ridge_dataloader()
    # "./results/Pavia/data/"           pavia_dataloader()
    # "./results/Salinas/data/"         salinas_dataloader()
    # "./results/SalinasA/data/"        salinas_a_dataloader()
    # "./results/Samson/data/"          samson_dataloader()
    # "./result/tests/data"           test_dataloader()
    '''

    # file_spectral = "./results/JasperRidge/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    # file_labels = "./results/JasperRidge/data/clustering_kmeans_linear_autoencoder_1.txt"
    # single_analyse(jasper_ridge_dataloader(), file_spectral, file_labels)

    # file_spectral = "./results/IndianPines/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    # file_labels = "./results/IndianPines/data/clustering_kmeans_linear_autoencoder_1.txt"
    # single_analyse(indian_pines_dataloader(), file_spectral, file_labels)

    # file_spectral = "./results/Samson/data/spectral_curve_clustering_kmeans_linear_autoencoder_1.txt"
    # file_labels = "./results/Samson/data/clustering_kmeans_linear_autoencoder_1.txt"
    # single_analyse(samson_dataloader(), file_spectral, file_labels)

    print("END")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        output_file.close()

    # plt.show()
    # to many images
