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
import os
import matplotlib.pyplot as plt
import sys

'''
    Main function
'''

from models.autoencoder_linear_1 import Autoencoder as Autoencoder1
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


def cut_out_in_midle(the_image_autoencoded, Dataloader, verbal=False):
    result = np.copy(the_image_autoencoded)
    labels = Dataloader.get_labels(verbal=False)

    if verbal:
        print()
        print("***   Cutting out background   ***")
        print("---------------------------------")

    iterator = 0
    for row in range(labels.shape[0]):
        for column in range(labels.shape[1]):
            if labels[row][column] == Dataloader.background_label:
                result[iterator] = np.zeros((result.shape[1]))
            iterator += 1

    return result


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
        Autoencoder, Dataloader, Classifier, nr_of_clusters,
        show_img=True, save_img=True, save_data=True,
        first=False,
        middle_cut_out=False):

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

    dataloader_suffix = ""
    if "cut_out" in Dataloader.get_name():
        dataloader_suffix = "_cu"

    autoencoder_learned_file = \
        Dataloader.get_results_directory() + \
        'autoencoder/' + my_net.getType() + \
        '_' + my_net.getName() + dataloader_suffix + '.pth'
    autoencoder_learned_file_description = \
        Dataloader.get_results_directory() + \
        'autoencoder/' + \
        my_net.getType() + \
        '_' + my_net.getName() + \
        dataloader_suffix + \
        '_description.txt'
    print("Autoencoder learn file: ", autoencoder_learned_file)

    if not first and my_net.getName() != 'none':
        print()
        print("***   Checking if model file exists   ***")
        print("---------------------------------")
        if os.path.isfile(autoencoder_learned_file):
            print("\t File exists")
            print()
            print("***   Loading model from file   ***")
            print("---------------------------------")
            my_net.load_state_dict(torch.load(autoencoder_learned_file))
            my_net.eval()
        else:
            print("\t File doesnt exist")
            first = True

    if first and my_net.getName() != 'none':
        print()
        print("***   Creating optimizer   ***")
        print("---------------------------------")
        optimizer = torch.optim.Adam(my_net.parameters(), weight_decay=1e-5)
        criterion = nn.MSELoss()

        print()
        print("***   Learning   ***")
        print("---------------------------------")
        print("Is CUDA available: ", torch.cuda.is_available())
        from torch.autograd import Variable
        num_epochs = 10
        batch_size = 128
        learning_rate = 1e-3
        epsilon = 0.23
        the_best_loss = sys.maxsize
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

    print()
    print("***   Autoencoding immage   ***")
    print("---------------------------------")
    print("Image shape: ", the_image_shape)
    print("Image code got from autoencoder")
    the_image_autoencoded = [my_net.getCode(torch.Tensor(point)).detach().numpy() for point in the_image_list]
    print("Autoencoded image shape: ", np.shape(the_image_autoencoded))
    if middle_cut_out:
        print()
        print("***   Middle cut out   ***")
        print("---------------------------------")
        the_image_autoencoded = cut_out_in_midle(the_image_autoencoded, Dataloader)

    print()
    print("***   Clustering   ***")
    print("---------------------------------")
    the_image_classified = Classifier.clustering(the_image_autoencoded, the_image_shape, nr_of_clusters)
    print("Result shape: ", np.shape(the_image_classified))

    print()
    print("***   Printing   ***")
    print("---------------------------------")
    if show_img:
        plt.imshow(the_image_classified)
        plt.draw()
        # plt.show()

    print()
    print("***   Saving image   ***")
    print("---------------------------------")
    if save_img:
        plt.imshow(the_image_classified)
        suffix = ""
        if middle_cut_out:
            suffix = "_mcu"

        img_name = Classifier.get_name() + "_" + my_net.getType() + '_autoencoder_' + my_net.getName() + suffix + '.png'
        result_img_path = Dataloader.get_results_directory() + 'img/' + img_name
        print("Path: ", result_img_path)
        plt.savefig(result_img_path, bbox_inches='tight')

    print()
    print("***   Saving data   ***")
    print("---------------------------------")
    if save_data:
        data_name = Classifier.get_name() + "_" + my_net.getType() + '_autoencoder_' + my_net.getName() + '.txt'
        result_data_path = Dataloader.get_results_directory() + 'data/' + data_name
        print("Path: ", result_data_path)
        np.savetxt(result_data_path, the_image_classified, delimiter=" ", newline="\n", header=data_name, fmt="%s")


def run_machine_for_all():
    print()
    print("***   Run machine for all start   ***")
    print("=====================================")
    print()

    autoencoders = []
    autoencoders.append(Autoencoder1)
    autoencoders.append(Autoencoder2)
    autoencoders.append(Autoencoder3)
    # autoencoders.append(Autoencoder4)
    # autoencoders.append(Autoencoder5)

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
                print("========")
                print("Autoencoder : \t\t", Autoencoder.getType(), "\t", Autoencoder.getName())
                print("Dataloader:   \t\t", Dataloader().get_name())
                print("Clustering:   \t\t", clustring.get_name())
                print("========")
                run_machine(Autoencoder, Dataloader(), clustring, Dataloader.get_number_of_clusters()
                            , first=False, middle_cut_out=True)
                run_machine(Autoencoder, Dataloader(), clustring, Dataloader.get_number_of_clusters()
                            , first=False, middle_cut_out=False)

    print()
    print("***   Run machine for all end   ***")
    print("=====================================")
    print()


if __name__ == '__main__':
    to_file = False

    # Przekierowanie wyjścia do pliku
    if to_file:
        import sys
        orig_stdout = sys.stdout
        output_file = open('results/creator_labeled_image_output.txt', 'w')
        sys.stdout = output_file

    print("START")
    start_time = time.time()
    print("Start time:  ", time.ctime(start_time))

    # Procedures
    # run_machine_for_all()

    run_machine(
        Autoencoder1, Dataloader2(), classifier1, Dataloader2.get_number_of_clusters(),
        first=False, middle_cut_out=False)

    # end procedures

    print("\nEND")
    end_time = time.time()
    print("End time:  ", time.ctime(end_time))
    print("Duration:  ", int(end_time - start_time), " seconds")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        output_file.close()

    plt.show()

