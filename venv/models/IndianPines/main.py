#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
#from numpy.testing.tests.test_utils import my_cacw
from scipy import io
import numpy as np
import mathematical_operations as mo
import time

from autoencoder import Autoencoder
from clustering import clustering_kmeans

'''
    Indian Pines
    Result image does not mach labeled image.
'''

data_dir = 'C:/Users/Public/AI/artificial-intelligence---my-beginning/venv/data/Indian Pines/'

img_dir = './img/'

def run_machine():
    to_file = False

    # Przekierowanie wyjścia do pliku
    if to_file:
        import sys
        orig_stdout = sys.stdout
        output_file = open('output_file.txt', 'w')
        sys.stdout = output_file

    print("START")
    start_time = time.time()
    print("Start time:  ", time.ctime(start_time))

    print()
    print("***   Loading data   ***")
    print("---------------------------------")
    filename = 'Indian_pines_corrected.mat'
    ImDict = io.loadmat(data_dir + filename)
    image_name = 'indian_pines_corrected'
    the_image = ImDict[image_name]
    image_size = np.shape(the_image)
    NRows = image_size[0]
    NCols = image_size[1]
    NBands = image_size[2]
    print("Lokalizacja obrazu: \t", data_dir + filename)
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
    filename_labels = 'Indian_pines_gt.mat'
    ImDict_labels = io.loadmat(data_dir + filename_labels)
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
    print("Number of elements in dataset: ", my_dataset.__len__())

    print()
    print("***   Creating autoencoder   ***")
    print("---------------------------------")
    my_net = Autoencoder()
    print(my_net)
    params = list(my_net.parameters())
    print("Params size:  ", params.__len__())
    for parameter in params:
        print(len(parameter))

    print()
    print("***   Creating optimizer   ***")
    print("---------------------------------")
    optimizer = torch.optim.Adam(my_net.parameters(), weight_decay=1e-5)
    criterion = nn.MSELoss()

    print()
    print("***   Learning   ***")
    print("---------------------------------")
    from torch.autograd import Variable
    num_epochs = 3
    batch_size = 128
    learning_rate = 1e-3
    for epoch in range(num_epochs):
        for i, data in enumerate(my_dataloader):
            img, _ = data
            img = Variable(img).cpu()
            # ===================forward=====================
            output = my_net(img)
            loss = criterion(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print(type(loss))
        #print(len(loss))
        print(type(loss.item()))
        print('epoch [', epoch + 1, '/', num_epochs, '], loss:', loss.item())
        # if epoch % 10 == 0:
        #    pic = to_img(output.cpu().data)

    print()
    print("***   Saving model to file   ***")
    print("---------------------------------")
    autoencoder_learned_file = './autoencoder_' + my_net.getType + my_net.getName + '.pth'
    torch.save(my_net.state_dict(), autoencoder_learned_file)

    # print()
    # print("***   Loading model from file   ***")
    # print("---------------------------------")
    # autoencoder_learned_file = './autoencoder_' + my_net.getType + my_net.getName + '.pth'
    # my_net.load_state_dict(torch.load(autoencoder_learned_file))
    # my_net.eval()

    print()
    print("***   Checking code for one element   ***")
    print("---------------------------------")
    print(my_net.getCode(list_of_tensors[0]))

    print()
    print("***   Clustering   ***")
    print("---------------------------------")
    print()

    img_name = 'clustering_kmeans_' + my_net.getType() + '_autoencoder_' + my_net.getName() + '.png'
    clustering_kmeans(the_image, my_net, the_image_labels, img_dir, img_name)

    print("\nEND")
    end_time = time.time()
    print("End time:  ", time.ctime(end_time))
    print("Duration:  ", int(end_time - start_time), " seconds")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        output_file.close()


if __name__ == '__main__':
    run_machine()
