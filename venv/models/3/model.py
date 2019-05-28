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
from datetime import datetime


'''
    My first try with PyTorch based on "A 60 minute blitz"
    jasperRidge2_R198.mat
    
    
    Po klasyfikacji k - średnich pewien procent zostanie przekazany jako 
    niezaklasyfiowany.
    
    Będą to elementy leżące najdalej od średnich
'''

dir = '/home/michalhuras/Pulpit/artificial-intelligence---my-beginning-master' \
      '/venv/'
# dir = 'C:/TestingCatalog/AI_data/Indian Pines/Indian_pines_corrected.mat'

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
    print("***   K - means clastering   ***")
    print("---------------------------------")
    # https://www.datacamp.com/community/tutorials/k-means-clustering-python
    # https: // datatofish.com / k - means - clustering - python /

    from pandas import DataFrame
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    print("Image shape: ", np.shape(the_image))
    the_image_list = the_image
    # the_image_list = []
    # for row in the_image:
    #     for element in row:
    #         the_image_list.append(element)
    # print("List of points shape: ", np.shape(the_image_list))

    print("Image code got from autoencoder")
    image_autoencoded = [my_net.getCode(torch.Tensor(point)).detach().numpy()
                         for point in the_image_list]

    print("Creating dataframe from k-clastering")
    df = DataFrame(data=image_autoencoded)

    print("KMeans clastering")
    for nr_of_clasters in range(1, 31):
        print("Number of claters: ", nr_of_clasters)
        # number_of_clusters = 10
        number_of_clusters = nr_of_clasters
        kmeans = KMeans(n_clusters=number_of_clusters).fit(df)

        print("Creating list for clastered data")
        # clastered_data = np.zeros(np.shape(the_image_labels))
        clastered_data = np.zeros((100, 100))

        print("Clustered data shape:  ", np.shape(clastered_data))

        x = 0
        y = 0
        for i in range(np.shape(clastered_data)[0] * np.shape(clastered_data)[1]):
            clastered_data[x][y] = kmeans.predict(
                    [image_autoencoded[y * 100 + x]])
            x = x + 1
            if x == 100:
                x = 0
                y = y + 1

        if True:
            cluster_centers = kmeans.cluster_centers_
            print("Cluster centers:  ", cluster_centers)
            print(cluster_centers[0])

            nr_elements = np.zeros(nr_of_clasters + 1)

            x = 0
            y = 0
            for i in range(100*100):
                # print(int(clastered_data[x][y]))
                nr_elements[int(clastered_data[x][y])] = \
                   nr_elements[int(clastered_data[x][y])] + 1
                x = x + 1
                if x == 100:
                    x = 0
                    y = y + 1

            print("Number of elements in clusters:  ", nr_elements)

            my_range = 0.2
            nr_elements =\
                my_range * \
                np.shape(clastered_data)[0] * \
                np.shape(clastered_data)[1]

            points_distance = np.zeros(np.shape(clastered_data))

            def count_distance(a, b):
                dystans = 0
                for ele in range(len(a)):
                    dystans = dystans + ((a[ele] - b[ele])**2)
                import math
                dystans = math.sqrt(dystans)
                return dystans


            x1 = 0
            y1 = 0
            for i in range(100 * 100):
                points_distance[x1][y1] = count_distance(
                            image_autoencoded[y1 * 100 + x1],
                            cluster_centers[int(clastered_data[x1][y1])])
                x1 = x1 + 1
                if x1 == 100:
                    x1 = 0
                    y1 = y1 + 1

            plt.imshow(points_distance)
            name = 'dist' + str(nr_of_clasters) + '.png'
            plt.savefig(name, bbox_inches='tight')
            plt.show()





        import matplotlib.pyplot as plt

        # print(clastered_data)
        plt.imshow(clastered_data)
        name = 'img_clasters_' + str(nr_of_clasters) + '.png'
        plt.savefig(name, bbox_inches='tight')
        # plt.show()

    print()
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("\nEND")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        f.close()


