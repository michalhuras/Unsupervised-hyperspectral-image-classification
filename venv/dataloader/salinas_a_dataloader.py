#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from scipy import io
import numpy as np
import scripts_my.mathematical_operations as mo
import time


'''
    Salinas A
'''

g_nr_of_clusters = 7


class Dataloader:
    def __init__(self):
        self.data_dir = 'C:/Users/Public/AI/artificial-intelligence---my-beginning/venv/data/SalinasA/'
        # self.data_dir = './data/SalinasA/'
        self.results_dir = './results/SalinasA/'
        self.name = 'salinas_a'

        self.image_shape = (83, 86, 204)
        self.nr_of_clusters = g_nr_of_clusters
        self.image = ()
        self.image_list = ()
        self.image_labels = ()

        self.image_exists = False
        self.image_list_exists = False
        self.image_labels_exists = False

        self.background_label = 1

    def get_name(self, verbal=False):
        if verbal:
            print()
            print("***   Get name   ***")
            print("---------------------------------")

        return self.name

    def get_results_directory(self, verbal=False):
        if verbal:
            print()
            print("***   Get results directory   ***")
            print("---------------------------------")

        return self.results_dir

    @staticmethod
    def get_number_of_clusters(verbal=False):
        if verbal:
            print()
            print("***   Get number of clusters   ***")
            print("---------------------------------")

        return g_nr_of_clusters

    def get_image(self, verbal=False):
        if verbal:
            print()
            print("***   Get image   ***")
            print("---------------------------------")

        if not self.image_exists:
            filename = 'SalinasA_corrected.mat'
            ImDict = io.loadmat(self.data_dir + filename)
            image_name = 'salinasA_corrected'
            the_image = ImDict[image_name]
            image_size = np.shape(the_image)
            NRows = image_size[0]
            NCols = image_size[1]
            depth = image_size[2]
            print("Lokalizacja obrazu: \t", self.data_dir + filename)
            print("Nazwa obrazu:  \t\t\t", image_name)
            print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols, " głębokość: ", depth)
            print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

            print()
            print("***   Converting image to uint8   ***")
            print("---------------------------------")
            the_image = mo.numpy_to_uint8(the_image)

            self.image = the_image
            self.image_exists = True

        return self.image

    def get_image_list(self, verbal=False):
        if verbal:
            print()
            print("***   Get image list   ***")
            print("---------------------------------")

        if not self.image_list_exists:
            the_image = self.get_image(False)
            the_image_list = np.reshape(the_image, (self.image_shape[0] * self.image_shape[1], self.image_shape[2]))

            image_size = np.shape(the_image_list)
            length = image_size[0]
            depth = image_size[1]
            if verbal:
                print("Rozmiar: \t\t\t\t", "długość: ", length, " głębokość: ", depth)
                print("Ilośc pikseli (długość * głębokość): ", length * depth)
            self.image_list = the_image_list
            self.image_list_exists = True

        return self.image_list

    def get_image_shape(self, verbal=False):
        if verbal:
            print()
            print("***   Getting shape   ***")
            print("---------------------------------")

        return self.image_shape

    def get_labels(self, verbal=False):
        if verbal:
            print()
            print("***   Loading labels   ***")
            print("---------------------------------")

        if not self.image_labels_exists:
            # To juz jest w uint8
            filename_labels = 'SalinasA_gt.mat'
            ImDict_labels = io.loadmat(self.data_dir + filename_labels)
            image_name_labels = 'salinasA_gt'
            the_image_labels = ImDict_labels[image_name_labels]


            image_size_labels = np.shape(the_image_labels)
            NRows_labels = image_size_labels[0]
            NCols_labels = image_size_labels[1]

            # labels unification - wartości od 0 do number_of_labels -1
            unused_label = 0
            labels_dictionary = {}
            x = 0
            y = 0
            labels_values = set()
            for i in range(self.image_shape[0] * self.image_shape[1]):
                if the_image_labels[y, x] not in labels_dictionary:
                    labels_dictionary[the_image_labels[y, x]] = unused_label
                    unused_label += 1
                the_image_labels[y, x] = labels_dictionary[the_image_labels[y, x]]
                labels_values.add(the_image_labels[y, x])
                x = x + 1
                if x == self.image_shape[1]:
                    x = 0
                    y += 1

            # import matplotlib.pyplot as plt
            # plt.imshow(the_image_labels)
            # plt.show()

            if verbal:
                print("Lokalizacja obrazu: \t", filename_labels)
                print("Nazwa obrazu:  \t\t\t", image_name_labels)
                print("Rozmiar: \t\t\t\t", "wiersze: ", NRows_labels, " kolumny: ", NCols_labels)
                print("Ilośc etykiet: \t\t\t", self.nr_of_clusters)
                print("Etykiety: \t\t\t\t", labels_values)
            self.image_labels = the_image_labels
            self.image_labels_exists = True

        return self.image_labels

    def get_dataloader(self, verbal=False):
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
