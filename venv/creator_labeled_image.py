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
    Main function
'''

from models.autoencoder_linear import Autoencoder as Autoencoder1
from models.autoencoder_linear_2 import Autoencoder as Autoencoder2
from models.autoencoder_linear_3 import Autoencoder as Autoencoder3
# from models.autoencoder_none import Autoencoder as Autoencoder4
# from models.autoencoder_convolutional import Autoencoder as Autoencoder5

from dataloader.indian_pines_dataloader import Dataloader as Dataloader1
from dataloader.indian_pines_cut_out_dataloader import Dataloader as Dataloader11
from dataloader.jasper_ridge_dataloader import Dataloader as Dataloader2
from dataloader.pavia_dataloader import Dataloader as Dataloader3
from dataloader.pavia_cut_out_dataloader import Dataloader as Dataloader33
from dataloader.salinas_dataloader import Dataloader as Dataloader4
from dataloader.salinas_cut_out_dataloader import Dataloader as Dataloader44
from dataloader.salinas_a_dataloader import Dataloader as Dataloader5
from dataloader.salinas_a_cut_out_dataloader import Dataloader as Dataloader55
from dataloader.samson_dataloader import Dataloader as Dataloader6

import clustering.kmeans as classifier1
# import clustering.optics as classifier2
# import clustering.mean_shift as classifier3
# import clustering.gaussian_mixture as classifier4


def save_model(my_net, autoencoder_learned_file, autoencoder_learned_file_description, loss_value, dataset):
    print()
    print("***   Saving model to file   ***")
    print("---------------------------------")
    torch.save(my_net.state_dict(), autoencoder_learned_file)
    description_file = open(autoencoder_learned_file_description, "w+")
    description_file.write("Autoencoder: \n")
    description_file.write("Type: " + my_net.getType() + "\n")
    description_file.write("Name: " + my_net.getName() + "\n")
    description_file.write("Loss value: " + str(loss_value) + "\n")
    description_file.write("Dataset: " + dataset + "\n")
    description_file.write(str(my_net))
    description_file.write("\n")
    params = list(my_net.parameters())
    description_file.write("Params length:  " + str(params.__len__()))
    description_file.write("\n")
    for parameter in params:
        description_file.write("   " + str(len(parameter)))

    description_file.close()


def run_machine(
        Autoencoder, Dataloader, Classifier, nr_of_clusters, show_img=True, save_img=True, save_data=True, first=True):
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

    my_dataloader = Dataloader.get_dataloader()
    the_image_shape = Dataloader.get_image_shape()
    # (x, y, z)
    the_image_list = Dataloader.get_image_list()
    # shape: x * y, z

    print()
    print("***   Creating autoencoder   ***")
    print("---------------------------------")
    my_net = Autoencoder(the_image_shape[2])
    print(my_net)
    params = list(my_net.parameters())
    print("Params size:  ", params.__len__())
    for parameter in params:
        print(len(parameter))

    if my_net.getName() != 'none':
        print()
        print("***   Creating optimizer   ***")
        print("---------------------------------")
        optimizer = torch.optim.Adam(my_net.parameters(), weight_decay=1e-5)
        criterion = nn.MSELoss()

    autoencoder_learned_file = \
        Dataloader.get_results_directory() + '/autoencoder/' + my_net.getType() + '_' + my_net.getName() + '.pth'
    autoencoder_learned_file_description = \
        Dataloader.get_results_directory() + \
        '/autoencoder/' + \
        my_net.getType() + \
        '_' + my_net.getName() + \
        '_description.txt'
    if first and my_net.getName() != 'none':
        print()
        print("***   Learning   ***")
        print("---------------------------------")
        print("Is CUDA available: ", torch.cuda.is_available())
        from torch.autograd import Variable
        num_epochs = 10
        batch_size = 128
        learning_rate = 1e-3
        epsilon = 0.23
        the_best_loss = 1
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
            print('epoch [', epoch + 1, '/', num_epochs, '], loss:', loss.item())
            # if loss.item() < epsilon:
            #     print('Epsilon break. Epsilon value: ',epsilon)
            #     break
            if loss.item() < the_best_loss:
                the_best_loss = loss.item()
                save_model(
                    my_net,
                    autoencoder_learned_file,
                    autoencoder_learned_file_description,
                    the_best_loss,
                    Dataloader.get_name())

    if not first and my_net.getName() != 'none':
        print()
        print("***   Loading model from file   ***")
        print("---------------------------------")
        my_net.load_state_dict(torch.load(autoencoder_learned_file))
        my_net.eval()

    print()
    print("***   Autoencoding immage   ***")
    print("---------------------------------")
    print()

    print("Image shape: ", the_image_shape)
    print("Image code got from autoencoder")
    the_image_autoencoded = [my_net.getCode(torch.Tensor(point)).detach().numpy() for point in the_image_list]
    print("Autoencoded image shape: ", np.shape(the_image_autoencoded))

    print()
    print("***   Clustering   ***")
    print("---------------------------------")
    print()

    the_image_classified = Classifier.clustering(the_image_autoencoded, the_image_shape, nr_of_clusters)
    print("Result shape: ", np.shape(the_image_classified))

    print()
    print("***   Printing   ***")
    print("---------------------------------")
    print()
    import matplotlib.pyplot as plt
    if show_img:
        plt.imshow(the_image_classified)
        plt.show()

    print()
    print("***   Saving image   ***")
    print("---------------------------------")
    print()
    if save_img:
        plt.imshow(the_image_classified)
        img_name = Classifier.get_name() + "_" + my_net.getType() + '_autoencoder_' + my_net.getName() + '.png'
        result_img_path = Dataloader.get_results_directory() + 'img/' + img_name
        print("Path: ", result_img_path)
        plt.savefig(result_img_path, bbox_inches='tight')

    print()
    print("***   Saving data   ***")
    print("---------------------------------")
    print()
    if save_data:
        data_name = Classifier.get_name() + "_" + my_net.getType() + '_autoencoder_' + my_net.getName() + '.txt'
        result_data_path = Dataloader.get_results_directory() + 'data/' + data_name
        print("Path: ", result_data_path)
        np.savetxt(result_data_path, the_image_classified, delimiter=" ", newline="\n", header=data_name, fmt="%s")

    print("\nEND")
    end_time = time.time()
    print("End time:  ", time.ctime(end_time))
    print("Duration:  ", int(end_time - start_time), " seconds")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        output_file.close()

def run_machine_for_all():
    autoencoders = []
    autoencoders.append(Autoencoder1)
    autoencoders.append(Autoencoder2)
    autoencoders.append(Autoencoder3)

    dataloaders = []
    dataloaders.append(Dataloader1)
    dataloaders.append(Dataloader11)
    dataloaders.append(Dataloader2)
    dataloaders.append(Dataloader3)
    dataloaders.append(Dataloader33)
    dataloaders.append(Dataloader4)
    dataloaders.append(Dataloader44)
    dataloaders.append(Dataloader5)
    dataloaders.append(Dataloader55)
    dataloaders.append(Dataloader6)

    clustering_methods = []
    clustering_methods.append(classifier1)
    # clustering_methods.append(classifier2)
    # clustering_methods.append(classifier3)
    # clustering_methods.append(classifier4)

    for Autoencoder in autoencoders:
        for Dataloader in dataloaders:
            for clustring in clustering_methods:
                run_machine(Autoencoder, Dataloader(), clustring, Dataloader.get_number_of_clusters(), first=False)


if __name__ == '__main__':
    run_machine_for_all()

    #run_machine(Autoencoder1, Dataloader2(), classifier, Dataloader2.get_number_of_clusters(), first=True)
