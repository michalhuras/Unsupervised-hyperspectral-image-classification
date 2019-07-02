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


def clustering_kmeans(the_image, my_net, the_image_labels, img_dir, img_name, show_image=False):
    print()
    print("***   K - means clustering   ***")
    print("---------------------------------")
    # https://www.datacamp.com/community/tutorials/k-means-clustering-python
    # https: // datatofish.com / k - means - clustering - python /

    from pandas import DataFrame
    import matplotlib.pyplot as plt
    from sklearn.cluster import KMeans

    print("Image shape: ", np.shape(the_image))
    the_image_list = []
    for row in the_image:
        for element in row:
            the_image_list.append(element)
    print("List of points shape: ", np.shape(the_image_list))

    print("Image code got from autoencoder")
    image_autoencoded = [my_net.getCode(torch.Tensor(point)).detach().numpy() for point in the_image_list]

    print("Creating dataframe from k-clustering")
    df = DataFrame(data=image_autoencoded)

    print("KMeans clustering")
    number_of_clusters = 16
    kmeans = KMeans(n_clusters=number_of_clusters).fit(df)

    print("Creating list for clustered data")
    clastered_data = np.zeros(np.shape(the_image_labels))
    print("Clustered data shape:  ", np.shape(clastered_data))

    x = 0
    y = 0
    for i in range(np.shape(clastered_data)[0] * np.shape(clastered_data)[1]):
        # clustered_data[x][y] = kmeans.predict(image_autoencoded[y * 144 + x].reshape(1, -1))
        clastered_data[x][y] = kmeans.predict([image_autoencoded[y * 144 + x]])
        x = x + 1
        if x == 145:
            x = 0
            y = y + 1

    import matplotlib.pyplot as plt
    print(clastered_data)
    plt.imshow(clastered_data)
    name = img_dir + img_name
    plt.savefig(name, bbox_inches='tight')
    if show_image:
        plt.show()
