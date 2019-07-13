#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from scipy import io
import numpy as np
import time


'''
    Test dataloader
'''

g_nr_of_clusters = 3

class Dataloader():
    def __init__(self):
        self.data_dir = 'none'
        self.results_dir = './results/none/'
        self.name = 'test'

        self.image_shape = (6, 4, 5)
        self.nr_of_clusters = g_nr_of_clusters
        self.image = np.array
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
        if verbal:
            print()
            print("***   Get image   ***")
            print("---------------------------------")

        the_image = np.array(
                [[[0, 1, 1, 1, 0], [1, 2, 1, 3, 3], [1, 2, 1, 3, 3], [1, 2, 1, 3, 3]],
                [[0, 1, 1, 1, 0], [1, 2, 1, 3, 3], [1, 2, 1, 3, 3], [1, 2, 1, 3, 3]],
                [[0, 1, 1, 1, 0], [2, 3, 3, 4, 4], [2, 3, 3, 4, 4], [2, 3, 3, 4, 4]],
                [[0, 1, 1, 1, 0], [2, 3, 3, 4, 4], [2, 3, 3, 4, 4], [2, 3, 3, 4, 4]],
                [[0, 1, 1, 1, 0], [2, 3, 3, 4, 4], [2, 3, 3, 4, 4], [2, 3, 3, 4, 4]],
                [[0, 1, 1, 1, 0], [2, 3, 3, 4, 4], [2, 3, 3, 4, 4], [2, 3, 3, 4, 4]]])

        image_size = np.shape(the_image)
        NRows = image_size[0]
        NCols = image_size[1]
        depth = image_size[2]
        print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols, " głębokość: ", depth)
        print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

        self.image = the_image
        return self.image

    def get_image_list(self, verbal=True):
        if verbal:
            print()
            print("***   Get image list   ***")
            print("---------------------------------")

        the_imgae = self.get_image()

        the_image_list = \
            np.zeros((self.get_image_shape(False)[0] * self.get_image_shape(False)[1], self.get_image_shape(False)[2]))
        x = 0
        y = 0
        for i in range(self.get_image_shape(False)[0] * self.get_image_shape(False)[1]):
            the_image_list[i] = the_imgae[y, x]
            x += 1
            if x == self.get_image_shape(False)[1]:
                x = 0
                y += 1

        the_image_list_shape = np.shape(the_image_list)
        length = the_image_list_shape[0]
        depth = the_image_list_shape[1]

        if verbal:
            print("Rozmiar: \t\t\t\t", "długość: ", length, " głębokość: ", depth)
            print("Ilośc pikseli (długość * głębokość): ", length * depth)
        self.image_list = the_image_list

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

        the_image_labels = np.array(
                [[0, 1, 1, 1],
                [0, 1, 1, 1],
                [0, 2, 2, 2],
                [0, 2, 2, 2],
                [0, 2, 2, 2],
                [0, 2, 2, 2]])

        # import matplotlib.pyplot as plt
        # plt.imshow(the_image_labels)
        # plt.show()

        labels_sphape = the_image_labels.shape
        rows = labels_sphape[0]
        columns = labels_sphape[1]

        if verbal:
            print("Rozmiar: \t\t\t\t", "wiersze: ", rows, " kolumny: ", columns)
            print("Ilośc etykiet: \t\t\t", self.nr_of_clusters)
            print("Etykiety: \t\t\t\t", (0, 1, 2))
        self.image_labels = the_image_labels

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
