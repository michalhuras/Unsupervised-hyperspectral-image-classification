#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.insert(0, "/home/myname/pythonfiles")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as utils
import torchvision.transforms as transforms
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

dir = 'C:/TestingCatalog/AI_data/Indian Pines/'

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            # nn.Conv1d(200, 90, kernel_size=20, padding=1),
            # nn.ReLU(True),
            # nn.Conv1d(90, 40, kernel_size=20, padding=1),
            # nn.ReLU(True),
            # nn.Conv1d(40, 20, kernel_size=20, padding=1),
            # nn.ReLU(True))
            nn.Linear(200, 90),
            nn.Linear(90, 40),
            nn.Linear(40, 20))
        self.decoder = nn.Sequential(
            nn.Linear(20, 40),
            nn.Linear(40, 90),
            nn.Linear(90, 200))
            # nn.ConvTranspose1d(200, 90, kernel_size=20, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose1d(90, 40, kernel_size=20, padding=1),
            # nn.ReLU(True),
            # nn.ConvTranspose1d(40, 20, kernel_size=20, padding=1),
            # nn.ReLU(True),
            # nn.Sigmoid())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def getCode(self, x):
        # print(type(x))
        # print(len(x))
        # print(x)
        # print()
        # print()
        # print()
        x = self.encoder(x)
        # print(type(x))
        # print(len(x))
        # print(x)
        return x



if __name__ == '__main__':
    to_file = True

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
    filename = 'Indian_pines_corrected.mat'
    ImDict = io.loadmat(dir + filename)
    image_name = 'indian_pines_corrected'
    the_image = ImDict[image_name]
    image_size = np.shape(the_image)
    NRows = image_size[0]
    NCols = image_size[1]
    NBands = image_size[2]
    print("Lokalizacja obrazu: \t", dir + filename)
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
    ImDict_labels = io.loadmat(dir + filename_labels)
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
    # torch.save(my_net.state_dict(), './conv_autoencoder.pth')

    print()
    print("***   Loading model from file   ***")
    print("---------------------------------")
    my_net.load_state_dict(torch.load('./conv_autoencoder.pth'))
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

    print("all_points_200")
    all_points_200 = []
    for row in the_image:
        for element in row:
            all_points_200.append(element)

    print("autoencoded_points_20")
    autoencoded_points_20 = []
    for point in all_points_200:
        autoencoded_points_20.append(
                my_net.getCode(torch.Tensor(point)).detach().numpy())

    print("Data frame")
    df = DataFrame(data = autoencoded_points_20)

    print("kmeans")
    kmeans = KMeans(n_clusters = 16).fit(df)

    print("Zeros to labeled data")
    MYDATA = np.zeros(np.shape(the_image_labels))
    print(np.shape(MYDATA))


    x = 0
    y = 0
    for i in range(len(autoencoded_points_20)):
        MYDATA[y][x] = kmeans.predict(autoencoded_points_20[x * 144 +
                                                            y].reshape(1, -1))
        y = y + 1
        if y == 145:
            y = 0
            x = x + 1

    import matplotlib.pyplot as plt
    print(MYDATA)
    plt.imshow(MYDATA)
    plt.savefig('demo.png', bbox_inches='tight')
    # plt.show()

    print("\nEND")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        f.close()


