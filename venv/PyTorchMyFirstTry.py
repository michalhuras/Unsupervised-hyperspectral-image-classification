#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from torch.autograd import Variable
from scipy import io
import numpy as np
import mathematical_operations as mo


'''
    My first try with PyTorch based on "A 60 minute blitz"
    NOPE NOPE NOPE 
    not yet
    
    TODO 
        - sprawdzić jaka jest najlepsza wartość kernel_size
'''


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(20, 40, kernel_size=20),
            nn.ReLU(True),
            nn.Conv1d(40, 90, kernel_size=20),
            nn.ReLU(True),
            nn.Conv1d(90, 200, kernel_size=20),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(200, 90, kernel_size=20),
            nn.ReLU(True),
            nn.ConvTranspose1d(90, 40, kernel_size=20),
            nn.ReLU(True),
            nn.ConvTranspose1d(40, 20, kernel_size=20),
            nn.ReLU(True),
            nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


if __name__ == '__main__':
    to_file = False

    # Przekierowanie wyjścia do pliku
    if to_file:
        import sys
        orig_stdout = sys.stdout
        f = open('out.txt', 'w')
        sys.stdout = f

    print("START")

    print()
    print("***   Loading data   ***")
    print("---------------------------------")
    filename = 'C:/TestingCatalog/AI_data/Indian Pines/Indian_pines_corrected.mat'
    ImDict = io.loadmat(filename)
    image_name = 'indian_pines_corrected'
    the_image = ImDict[image_name]
    image_size = np.shape(the_image)
    NRows = image_size[0]
    NCols = image_size[1]
    NBands = image_size[2]
    print("Lokalizacja obrazu: \t", filename)
    print("Nazwa obrazu:  \t\t\t", image_name)
    print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols, " zakresy: ", NBands)
    print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

    print()
    print("***   Converting image to uint8   ***")
    print("---------------------------------")
    # converted_image = mo.numpy_to_uint8(the_image)
    the_image = mo.numpy_to_uint8(the_image)

    print()
    print("***   Loading labels   ***")
    print("---------------------------------")
    # To juz jest w uint8
    filename_labels = 'C:/TestingCatalog/AI_data/Indian Pines/Indian_pines_gt.mat'
    ImDict_labels = io.loadmat(filename_labels)
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
    print("Lokalizacja obrazu: \t", filename_labels)
    print("Nazwa obrazu:  \t\t\t", image_name_labels)
    print("Rozmiar: \t\t\t\t", "wiersze: ", NRows_labels, " kolumny: ", NCols_labels)
    print("Ilośc etykiet: ", num_labels, " Etykiety: ", labels)

    print()
    print("***   Creating datasets and dataloaders   ***")
    print("---------------------------------")

    list_of_tensors = []
    for row in the_image:
        for element in row:
            list_of_tensors.append(torch.Tensor(element))
    print(len(list_of_tensors))
    print(list_of_tensors[0])

    list_of_tensors_labels = []
    sie_xxx = 0
    for row in the_image_labels:
        for element in row:
            list_of_tensors_labels.append(torch.Tensor([element]))
    print(len(list_of_tensors_labels))
    print(list_of_tensors_labels[1])

    my_tensor = torch.stack(list_of_tensors)
    my_tensor_result = torch.stack(list_of_tensors_labels)
    my_dataset = utils.TensorDataset(my_tensor, my_tensor_result)
    my_dataloader = utils.DataLoader(my_dataset, batch_size=4)
    print("Number of elements in dataset: ", my_dataset.__len__())

    # list_of_tensors = []
    # for row in the_image:
    #     for element in row:
    #         list_of_tensors.append(torch.Tensor(element))
    #
    # train_size = int(0.8 * len(list_of_tensors))
    # test_size = len(list_of_tensors) - train_size
    # my_tensor = torch.stack(list_of_tensors)
    # print(type(my_tensor))
    # my_dataset = utils.TensorDataset(my_tensor)
    # print(type(my_dataset))
    # train_dataset, test_dataset = utils.dataset.random_split(my_dataset, [train_size, test_size])
    # print(type(train_dataset))
    # print(type(test_dataset))
    # train_dataset = utils.TensorDataset(train_dataset)
    # test_dataset = utils.TensorDataset(test_dataset)
    # train_dataloader = utils.DataLoader(train_dataset)
    # print(type(train_dataloader))
    # test_dataloader = utils.DataLoader(test_dataset)
    # print(type(test_dataloader))
    # print("Full dataset length: ", len(list_of_tensors))
    # print("Number of elements in train dataset: ", train_dataset.__len__())
    # print("Number of elements in test dataset: ", test_dataset.__len__())

    print()
    print("***   Creating autoencoder   ***")
    print("---------------------------------")
    my_net = Autoencoder().cpu()
    print(my_net)
    params = list(my_net.parameters())
    print(len(params))
    for parameter in params:
        print(parameter.size())

    distance = nn.MSELoss()

    print()
    print("***   Creating optimizer   ***")
    print("---------------------------------")
    optimizer = torch.optim.Adam(my_net.parameters(), weight_decay=1e-5)

    print()
    print("***   Training the autoencoder   ***")
    print("---------------------------------")
    num_epochs = 5
    batch_size = 128
    for epoch in range(num_epochs):
        for data in my_dataloader:
            print(type(data))
            print(len(data))
            img, _ = data
            print(type(img))
            print(len(img))
            print()
            # img = Variable(img).cpu()
            # ===================forward=====================
            output = my_net(img)
            loss = distance(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss))

    print("\nEND")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        f.close()


