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
    Indian Pines
    TODO dodać obraz do pamięci, tak aby przy każdym wywołaniu funkcji nie trzeba było go na nowo odczytywać
    if notloaded .. 
'''

g_nr_of_clusters = 20

class Dataloader():
    def __init__(self):
        self.data_dir = 'C:/Users/Public/AI/artificial-intelligence---my-beginning/venv/data/Indian Pines/'
        self.results_dir = './results/IndianPines/'
        self.name = 'indian_pines'


        self.nr_of_clusters = g_nr_of_clusters

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
            print("***   Loading data   ***")
            print("---------------------------------")
        filename = 'Indian_pines_corrected.mat'
        ImDict = io.loadmat(self.data_dir + filename)
        image_name = 'indian_pines_corrected'
        the_image = ImDict[image_name]
        image_size = np.shape(the_image)
        NRows = image_size[0]
        NCols = image_size[1]
        NBands = image_size[2]
        if verbal:
            print("Lokalizacja obrazu: \t", self.data_dir + filename)
            print("Nazwa obrazu:  \t\t\t", image_name)
            print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols, " zakresy: ", NBands)
            print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

        if verbal:
            print()
            print("***   Converting image to uint8   ***")
            print("---------------------------------")
            # converted_image = mo.numpy_to_uint8(the_image)
        the_image = mo.numpy_to_uint8(the_image)

        return the_image

    def get_image_list(self, verbal=True):
        if verbal:
            print()
            print("***   Get image list   ***")
            print("---------------------------------")
        filename = 'Indian_pines_corrected.mat'
        ImDict = io.loadmat(self.data_dir + filename)
        image_name = 'indian_pines_corrected'
        the_image = ImDict[image_name]
        image_size = np.shape(the_image)
        NRows = image_size[0]
        NCols = image_size[1]
        NBands = image_size[2]
        if verbal:
            print("Lokalizacja obrazu: \t", self.data_dir + filename)
            print("Nazwa obrazu:  \t\t\t", image_name)
            print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols, " zakresy: ", NBands)
            print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

        if verbal:
            print()
            print("***   Converting image to uint8   ***")
            print("---------------------------------")
            # converted_image = mo.numpy_to_uint8(the_image)
        the_image = mo.numpy_to_uint8(the_image)

        the_image_list = []
        for row in the_image:
            for element in row:
                the_image_list.append(element)
        print("List of points shape: ", np.shape(the_image_list))

        return the_image_list

    def get_image_shape(self, verbal=True):
        if verbal:
            print()
            print("***   Getting shape   ***")
            print("---------------------------------")
        filename = 'Indian_pines_corrected.mat'
        ImDict = io.loadmat(self.data_dir + filename)
        image_name = 'indian_pines_corrected'
        the_image = ImDict[image_name]
        image_shape = np.shape(the_image)
        NRows = image_shape[0]
        NCols = image_shape[1]
        NBands = image_shape[2]
        if verbal:
            print("Lokalizacja obrazu: \t", self.data_dir + filename)
            print("Nazwa obrazu:  \t\t\t", image_name)
            print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols, " zakresy: ", NBands)
            print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

        return image_shape

    def get_labels(self, verbal=True):
        if verbal:
            print()
            print("***   Loading labels   ***")
            print("---------------------------------")
        # To juz jest w uint8
        filename_labels = 'Indian_pines_gt.mat'
        ImDict_labels = io.loadmat(self.data_dir + filename_labels)
        image_name_labels = 'indian_pines_gt'
        the_image_labels = ImDict_labels[image_name_labels]
        image_size_labels = np.shape(the_image_labels)
        NRows_labels = image_size_labels[0]
        NCols_labels = image_size_labels[1]
        '''
        import matplotlib.pyplot as plt
        plt.imshow(the_image_labels)
        plt.show()
        '''
        labels = set()
        for row in the_image_labels:
            for element in row:
                labels.add(element)
        num_labels = len(labels)
        if verbal:
            print("Lokalizacja obrazu: \t", filename_labels)
            print("Nazwa obrazu:  \t\t\t", image_name_labels)
            print("Rozmiar: \t\t\t\t", "wiersze: ", NRows_labels, " kolumny: ", NCols_labels)
            print("Ilośc etykiet: ", num_labels, " Etykiety: ", labels)

        return the_image_labels

    def get_dataloader(self, verbal=True):
        the_image = self.get_image(False)
        the_image_labels = self.get_labels(False)

        if verbal:
            print()
            print("***   Creating dataset and dataloader   ***")
            print("---------------------------------")
        import torch.utils.data as utils
        list_of_tensors = []
        for row in the_image:
            for element in row:
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
