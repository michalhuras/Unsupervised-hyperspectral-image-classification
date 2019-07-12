#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from scipy import io
import numpy as np
import mathematical_operations as mo
import time


'''
    Jasper ridge
'''

g_nr_of_clusters = 4

class Dataloader():
    def __init__(self):
        self.data_dir = 'C:/Users/Public/AI/artificial-intelligence---my-beginning/venv/data/Jasper_ridge/'
        # self.data_dir = './data/Jasper_ridge/'
        self.results_dir = './results/JasperRidge/'
        self.name = 'jasper_ridge'

        self.image_shape = (100, 100, 198)
        self.nr_of_clusters = g_nr_of_clusters
        self.image = ()
        self.image_list = ()
        self.image_labels = ()

        self.image_exists = False
        self.image_list_exists = False
        self.image_labels_exists = False

    def get_name(self, verbal=True):
        if verbal:
            print()
            print("***   Get name   ***")
            print("---------------------------------")

        return self.name

    def get_results_directory(self, verbal=True):
        if verbal:
            print()
            print("***   Get results directory   ***")
            print("---------------------------------")

        return self.results_dir

    @staticmethod
    def get_number_of_clusters(verbal=True):
        if verbal:
            print()
            print("***   Get number of clusters   ***")
            print("---------------------------------")

        return g_nr_of_clusters

    def get_image(self, verbal=True):
        if not self.image_exists:
            the_image_list = self.get_image_list(False)
            if verbal:
                print()
                print("***   Get image   ***")
                print("---------------------------------")

            the_image = np.reshape(the_image_list, self.image_shape)
            this_shape = np.shape(the_image)
            NRows = this_shape[0]
            NCols = this_shape[1]
            depth = this_shape[2]
            if verbal:
                print("Rozmiar: \t\t", "wiersze: ", NRows, " kolumny: ", NCols, " głębokośc: ", depth)
                print("Ilośc pikseli (ilość kolumn * ilość wierszy * głębokośc): ", NRows * NCols * depth)

            self.image_exists = True
            self.image = the_image

        return self.image


    def get_image_list(self, verbal=True):
        if verbal:
            print()
            print("***   Get image list   ***")
            print("---------------------------------")

        if not self.image_list_exists:
            filename = 'jasperRidge2_R198.mat'
            ImDict = io.loadmat(self.data_dir + filename)
            image_name = 'Y'
            the_image_list = ImDict[image_name]
            the_image_list = the_image_list.transpose()
            the_image_list_turned = mo.turn_image_in_list(the_image_list, self.image_shape)
            image_size = np.shape(the_image_list_turned)
            NRows = image_size[0]
            NCols = image_size[1]
            print("Lokalizacja obrazu: \t", self.data_dir + filename)
            print("Nazwa obrazu:  \t\t\t", image_name)
            print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols)
            print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

            print()
            print("***   Converting image to uint8   ***")
            print("---------------------------------")
            the_image_list = mo.numpy_to_uint8(the_image_list)

            self.image_list = the_image_list_turned
            self.image_list_exists = True

        return self.image_list

    def get_image_shape(self, verbal=True):
        if verbal:
            print()
            print("***   Getting shape   ***")
            print("---------------------------------")

        return self.image_shape

    def get_labels(self, verbal=True):
        if verbal:
            print()
            print("***   Loading labels   ***")
            print("---------------------------------")

        if not self.image_labels_exists:
            # To juz jest w uint8
            filename_labels = 'GroundTruth/end4.mat'
            ImDict_labels = io.loadmat(self.data_dir + filename_labels)
            image_name_labels = 'A'
            the_image_labels = ImDict_labels[image_name_labels]
            the_image_labels = the_image_labels.transpose()
            the_image_labels = np.reshape(the_image_labels, (self.image_shape[0], self.image_shape[1], 4))

            corrected_labels_1D = np.zeros((self.image_shape[0], self.image_shape[1]))
            for i, row in enumerate(corrected_labels_1D):
                for j, element in enumerate(row):
                    if (the_image_labels[i][j][0] > the_image_labels[i][j][1]) and \
                            (the_image_labels[i][j][0] > the_image_labels[i][j][2]) and \
                            (the_image_labels[i][j][0] > the_image_labels[i][j][3]):
                        corrected_labels_1D[j][i] = 0
                    elif (the_image_labels[i][j][1] > the_image_labels[i][j][0]) and \
                            (the_image_labels[i][j][1] > the_image_labels[i][j][2]) and \
                            (the_image_labels[i][j][1] > the_image_labels[i][j][3]):
                        corrected_labels_1D[j][i] = 1
                    elif (the_image_labels[i][j][2] > the_image_labels[i][j][0]) and \
                         (the_image_labels[i][j][2] > the_image_labels[i][j][1]) and \
                         (the_image_labels[i][j][2] > the_image_labels[i][j][3]):
                        corrected_labels_1D[j][i] = 2
                    else:
                        corrected_labels_1D[j][i] = 3

            # import matplotlib.pyplot as plt
            # plt.imshow(corrected_labels_1D)
            # plt.show()

            image_size_labels = np.shape(corrected_labels_1D)
            NRows_labels = image_size_labels[0]
            NCols_labels = image_size_labels[1]

            if verbal:
                print("Lokalizacja obrazu: \t", filename_labels)
                print("Nazwa obrazu:  \t\t\t", image_name_labels)
                print("Rozmiar: \t\t\t\t", "wiersze: ", NRows_labels, " kolumny: ", NCols_labels)
                print("Ilośc etykiet: \t\t\t", self.get_number_of_clusters())
                print("Etykiety: \t\t\t\t", (0, 1, 2, 3))
            self.image_labels = corrected_labels_1D
            self.image_labels_exists = True

        return self.image_labels

    def get_dataloader(self, verbal=True):
        the_image_list = self.get_image_list(False)
        the_image_labels = self.get_labels(False)

        if verbal:
            print()
            print("***   Creating dataset and dataloader   ***")
            print("---------------------------------")

        import torch.utils.data as utils
        list_of_tensors = []
        for element in the_image_list:
            list_of_tensors.append(torch.Tensor(element))

        list_of_tensors_labels = []
        for row in the_image_labels:
            for element in row:
                list_of_tensors_labels.append(torch.Tensor([element]))

        my_tensor = torch.stack(list_of_tensors)
        my_tensor_labels = torch.stack(list_of_tensors_labels)
        my_dataset = utils.TensorDataset(my_tensor, my_tensor_labels)
        my_dataloader = utils.DataLoader(my_dataset)

        if verbal:
            print("Number of elements in dataset: ", my_dataset.__len__())

        return my_dataloader


if __name__ == '__main__':
    my_dataloader = Dataloader()
    print("\nTEST GET NAME")
    print("RESULT:  ", my_dataloader.get_name())
    print("\nTEST GET results directory")
    print("RESULT:  ", my_dataloader.get_results_directory())
    print("\nTEST GET NUMBER OF CLUSTERS")
    print("RESULT:  ", my_dataloader.get_number_of_clusters())
    print("\nTEST GET IMAGE")
    print("RESULT:  ", np.shape(my_dataloader.get_image()))
    print("\nTEST GET IMAGE LIST")
    print("RESULT:  ", np.shape(my_dataloader.get_image_list()))
    print("\nTEST GET IMAGE SHAPE")
    print("RESULT:  ", my_dataloader.get_image_shape())
    print("\nTEST GET LABELS")
    print("RESULT:  ", np.shape(my_dataloader.get_labels()))
    print("\nTEST GET DATALOADER")
    print("RESULT:  ", my_dataloader.get_dataloader())
