#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
from pygments.formatters import img
from scipy import io
import numpy as np
import mathematical_operations as mo
from datetime import datetime


'''
    jasperRidge2_R198.mat
'''

dir = './../../data/Jasper_ridge/'

img_dir = "img/"

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(198, 90),
            nn.Linear(90, 40),
            nn.Linear(40, 20))
        self.decoder = nn.Sequential(
            nn.Linear(20, 40),
            nn.Linear(40, 90),
            nn.Linear(90, 198))

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def getCode(self, x):
        x = self.encoder(x)
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
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    print()
    print("***   Loading data   ***")
    print("---------------------------------")
    filename = 'jasperRidge2_R198.mat'
    ImDict = io.loadmat(dir + filename)
    image_name = 'Y'
    the_image = ImDict[image_name]
    the_image = the_image.transpose()
    image_size = np.shape(the_image)
    print(image_size)
    NRows = image_size[0]
    NCols = image_size[1]
    print("Lokalizacja obrazu: \t", dir + filename)
    print("Nazwa obrazu:  \t\t\t", image_name)
    print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols)
    print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

    print()
    print("***   Converting image to uint8   ***")
    print("---------------------------------")
    # converted_image = mo.numpy_to_uint8(the_image)
    the_image = mo.numpy_to_uint8(the_image)

    # print()
    # print("***   Loading labels   ***")
    # print("---------------------------------")
    # # To juz jest w uint8
    # filename_labels = 'Indian_pines_gt.mat'
    # ImDict_labels = io.loadmat(dir + filename_labels)
    # image_name_labels = 'indian_pines_gt'
    # the_image_labels = ImDict_labels[image_name_labels]
    # image_size_labels = np.shape(the_image_labels)
    # NRows_labels = image_size_labels[0]
    # NCols_labels = image_size_labels[1]
    # '''
    # import matplotlib.pyplot as plt
    # plt.imshow(the_image_labels)
    # plt.show()
    # '''
    # labels = set()
    # for row in the_image_labels:
    #     for element in row:
    #         labels.add(element)
    # num_labels = len(labels)
    # print("Lokalizacja obrazu: \t", filename_labels)
    # print("Nazwa obrazu:  \t\t\t", image_name_labels)
    # print("Rozmiar: \t\t\t\t", "wiersze: ", NRows_labels, " kolumny: ", NCols_labels)
    # print("Ilośc etykiet: ", num_labels, " Etykiety: ", labels)

    print()
    print("***   Creating dataset and dataloader   ***")
    print("---------------------------------")
    import torch.utils.data as utils

    list_of_tensors = []
    for element in the_image:
        list_of_tensors.append(torch.Tensor(element))

    # list_of_tensors_labels = []
    # for row in the_image_labels:
    #     for element in row:
    #         list_of_tensors_labels.append(torch.Tensor([element]))

    my_tensor = torch.stack(list_of_tensors)
    # my_tensor_labels = torch.stack(list_of_tensors_labels)
    my_dataset = utils.TensorDataset(my_tensor, my_tensor)
    # my_dataset = utils.TensorDataset(my_tensor, my_tensor_labels)
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

    # print()
    # print("***   Learning   ***")
    # print("---------------------------------")
    # from torch.autograd import Variable
    # num_epochs = 20
    # batch_size = 128
    # learning_rate = 1e-3
    # for epoch in range(num_epochs):
    #     for i, data in enumerate(my_dataloader):
    #         img, _ = data
    #         img = Variable(img).cpu()
    #         # ===================forward=====================
    #         output = my_net(img)
    #         loss = criterion(output, img)
    #         # ===================backward====================
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     # ===================log========================
    #     print(type(loss))
    #     print(type(loss.item()))
    #     print('epoch [', epoch + 1, '/', num_epochs, '], loss:', loss.item())
    #
    #
    # print()
    # print("***   Saving model to file   ***")
    # print("---------------------------------")
    # torch.save(my_net.state_dict(), './lin_autoencoder_20_epochs.pth')

    print()
    print("***   Loading model from file   ***")
    print("---------------------------------")
    my_net.load_state_dict(torch.load('./lin_autoencoder_20_epochs.pth'))
    my_net.eval()

    print()
    print("***   Testing for one element   ***")
    print("---------------------------------")
    print(my_net.getCode(list_of_tensors[0]))

    print()
    print("***   OPTICS clustering   ***")
    print("---------------------------------")
    # https://scikit-learn.org/stable/modules/clustering.html
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_optics.html
    # #sphx-glr-auto-examples-cluster-plot-optics-py
    # https://scikit-learn.org/stable/modules/clustering.html#optics
    import matplotlib.pyplot as plt
    from sklearn.cluster import OPTICS, cluster_optics_dbscan

    clust = OPTICS(min_samples=10, xi=.0005, min_cluster_size=.005)

    print("Image shape: ", np.shape(the_image))
    the_image_list = the_image

    print("Image code got from autoencoder")
    image_autoencoded = [my_net.getCode(torch.Tensor(point)).detach().numpy()
                         for point in the_image_list]

    print("Runing fit function for OPTICS clustering")
    clust.fit(image_autoencoded)

    labels_050 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=0.5)

    labels_200 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=2)

    labels_300 = cluster_optics_dbscan(reachability=clust.reachability_,
                                       core_distances=clust.core_distances_,
                                       ordering=clust.ordering_, eps=3)

    print("---------------------------")
    reachability = clust.reachability_[clust.ordering_]
    print(reachability)
    print("---------------------------")

    print("Creating list for clastered data")
    clustered_data = np.zeros((100, 100))
    clustered_data_labels_050 = np.zeros((100, 100))
    clustered_data_labels_200 = np.zeros((100, 100))
    clustered_data_labels_300 = np.zeros((100, 100))

    print("Clustered data shape:  ", np.shape(clustered_data))

    x = 0
    y = 0
    for i in range(np.shape(clustered_data)[0] * np.shape(clustered_data)[1]):
        clustered_data[x][y] = clust.labels_[i]
        clustered_data_labels_050[x][y] = labels_050[i]
        clustered_data_labels_200[x][y] = labels_200[i]
        clustered_data_labels_300[x][y] = labels_300[i]
        x = x + 1
        if x == 100:
            x = 0
            y = y + 1

    import matplotlib.pyplot as plt

    fig = plt.figure()
    plt.imshow(clustered_data)
    plt.title('Automatic Clustering\nOPTICS')
    name = 'img_OPTICS_clustering_automatic.png'
    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(clustered_data_labels_050)
    plt.title('Clustering at 0.5 epsilon cut\nDBSCAN')
    name = img_dir + 'img_OPTICS_clustering_0_5_clustering.png'
    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(clustered_data_labels_200)
    plt.title('Clustering at 2.0 epsilon cut\nDBSCAN')
    name = img_dir + 'img_OPTICS_clustering_2_0_epsilon.png'
    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.imshow(clustered_data_labels_300)
    plt.title('Clustering at 2.0 epsilon cut\nDBSCAN')
    name = img_dir + 'img_OPTICS_clustering_3_0_epsilon.png'
    plt.savefig(name, bbox_inches='tight')
    plt.close(fig)

    print()
    print("***   DBSCAN clustering   ***")
    print("---------------------------------")
    import matplotlib.pyplot as plt
    from sklearn.cluster import DBSCAN

    print("Image shape: ", np.shape(the_image))
    the_image_list = the_image

    print("Image code got from autoencoder")
    image_autoencoded = [my_net.getCode(torch.Tensor(point)).detach().numpy()
                         for point in the_image_list]

    print("Runing fit function for DBSCAN clustering")
    clust = DBSCAN(eps=3, min_samples=2).fit(image_autoencoded)

    print("Creating list for clastered data")
    clustered_data = np.zeros((100, 100))

    print("Clustered data shape:  ", np.shape(clustered_data))

    x = 0
    y = 0
    for i in range(np.shape(clustered_data)[0] * np.shape(clustered_data)[1]):
        clustered_data[x][y] = clust.labels_[i]
        x = x + 1
        if x == 100:
            x = 0
            y = y + 1

    import matplotlib.pyplot as plt

    plt.imshow(clustered_data)
    name = img_dir + 'img_DBSCAN_clustering.png'
    plt.savefig(name, bbox_inches='tight')

    print()
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("\nEND")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        f.close()


