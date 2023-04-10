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
    Tested for dataset Samson
    (the same model as in catalog 1, different dataset)
'''

data_dir = 'C:/Users/Public/AI/artificial-intelligence---my-beginning/venv/data/Samson/'

img_dir = "img/"

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(156, 75),
            nn.Linear(75, 35),
            nn.Linear(35, 15))
        self.decoder = nn.Sequential(
            nn.Linear(15, 35),
            nn.Linear(35, 75),
            nn.Linear(75, 156))

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
        output_file = open('output_file.txt', 'w')
        sys.stdout = output_file

    print("START")
    start_time = time.time()
    print("Start time:  ", time.ctime(start_time))

    print()
    print("***   Loading data   ***")
    print("---------------------------------")
    filename = 'samson_1.mat'
    ImDict = io.loadmat(data_dir + filename)
    image_name = 'V'
    the_image = ImDict[image_name]
    the_image = the_image.transpose()
    image_size = np.shape(the_image)
    print(image_size)
    NRows = image_size[0]
    NCols = image_size[1]
    # NBands = image_size[2]
    print("Lokalizacja obrazu: \t", data_dir + filename)
    print("Nazwa obrazu:  \t\t\t", image_name)
    print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols) #, " zakresy: ", NBands)
    print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)


    print()
    print("***   Converting image to uint8   ***")
    print("---------------------------------")
    # converted_image = mo.numpy_to_uint8(the_image)
    # the_image = mo.numpy_to_uint8(the_image)

    # print()
    # print("***   Loading labels   ***")
    # print("---------------------------------")
    # # To juz jest w uint8
    # filename_labels = 'GroundTruth/end3.mat'
    # ImDict_labels = io.loadmat(data_dir + filename_labels)
    # #print(ImDict_labels)
    # image_name_labels = 'M'
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
    # num_epochs = 3
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
    #     #print(len(loss))
    #     print(type(loss.item()))
    #     print('epoch [', epoch + 1, '/', num_epochs, '], loss:', loss.item())
    #     # if epoch % 10 == 0:
    #     #    pic = to_img(output.cpu().data)
    #
    # print()
    # print("***   Saving model to file   ***")
    # print("---------------------------------")
    # torch.save(my_net.state_dict(), './linear_autoencoder.pth')

    print()
    print("***   Loading model from file   ***")
    print("---------------------------------")
    my_net.load_state_dict(torch.load('./linear_autoencoder.pth'))
    my_net.eval()

    print()
    print("***   Checking code for one element   ***")
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
    image_autoencoded = [my_net.getCode(torch.Tensor(point)).detach().numpy() for point in the_image_list]

    print("Creating dataframe from k-clastering")
    df = DataFrame(data=image_autoencoded)

    print("KMeans clastering")
    for trololo in range(1, 10):
        print("Number of claters: ", trololo)
        # number_of_clusters = 10
        number_of_clusters = trololo
        kmeans = KMeans(n_clusters=number_of_clusters).fit(df)

        print("Creating list for clastered data")
        # clastered_data = np.zeros(np.shape(the_image_labels))
        clastered_data = np.zeros((95, 95))

        print("Clastered data shape:  ", np.shape(clastered_data))

        x = 0
        y = 0
        for i in range(np.shape(clastered_data)[0] * np.shape(clastered_data)[1]):
            clastered_data[x][y] = kmeans.predict([image_autoencoded[y * 95 + x]])
            x = x + 1
            if x == 95:
                x = 0
                y = y + 1

        import matplotlib.pyplot as plt
        print(clastered_data)
        plt.imshow(clastered_data)
        name = img_dir + 'img_clasters_' + str(trololo) + '.png'
        plt.savefig(name, bbox_inches='tight')
        # plt.show()

    print("\nEND")
    end_time = time.time()
    print("End time:  ", time.ctime(end_time))
    print("Duration:  ", int(end_time - start_time), " seconds")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        output_file.close()
