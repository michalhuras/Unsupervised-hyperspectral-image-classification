#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import sys
import csv

import creator_result_files as crf

'''
    Main function to create labeled images
'''

from models.autoencoder_none import Autoencoder as Autoencoder0
from models.autoencoder_linear_1 import Autoencoder as Autoencoder1
from models.autoencoder_linear_2 import Autoencoder as Autoencoder2
from models.autoencoder_linear_3 import Autoencoder as Autoencoder3
from models.autoencoder_linear_4 import Autoencoder as Autoencoder4
from models.autoencoder_linear_5 import Autoencoder as Autoencoder5
from models.autoencoder_linear_6 import Autoencoder as Autoencoder6
from models.autoencoder_convolutional_1 import Autoencoder as Autoencoder7
from models.autoencoder_convolutional_2 import Autoencoder as Autoencoder8
from models.autoencoder_convolutional_3 import Autoencoder as Autoencoder9

from dataloader.indian_pines_dataloader import Dataloader as Dataloader1
# from dataloader.indian_pines_cut_out_dataloader import Dataloader as Dataloader11
from dataloader.jasper_ridge_dataloader import Dataloader as Dataloader2
from dataloader.pavia_dataloader import Dataloader as Dataloader3
# from dataloader.pavia_cut_out_dataloader import Dataloader as Dataloader33
from dataloader.salinas_dataloader import Dataloader as Dataloader4
# from dataloader.salinas_cut_out_dataloader import Dataloader as Dataloader44
from dataloader.salinas_a_dataloader import Dataloader as Dataloader5
# from dataloader.salinas_a_cut_out_dataloader import Dataloader as Dataloader55
from dataloader.samson_dataloader import Dataloader as Dataloader6

import clustering.kmeans as classifier1
import clustering.gaussian_mixture as classifier2


def current_time_ms():
    # get current time in milliseconds
    return int(round(time.time() * 1000))


time_beg = 0
time_fin = 0


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


def save_csv_file(
        filename, row_to_write, verbal=True):
    print("save_model_file")
    with open(filename, 'a', newline='') as csvfile:
        if verbal:
            print("Filename: ", filename)
        file_writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        file_writer.writerow(row_to_write)


def run_machine(
        Autoencoder, Dataloader, Classifier, nr_of_clusters,
        show_img=True, save_img=True, save_data=True,
        first=False,
        middle_cut_out=False,
        param="",
        overwrite=False):
    print("========")
    print("Autoencoder : \t\t", Autoencoder.getType(), "\t", Autoencoder.getName())
    print("Dataloader:   \t\t", Dataloader.get_name())
    print("Clustering:   \t\t", Classifier.get_name())
    print("Param : \t\t", param)
    print("========")

    my_dataloader = Dataloader.get_dataloader(verbal=False)
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
        print(len(parameter), end="  ||  ")
    print()

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
    file_times_learning = \
        Dataloader.get_results_directory() + \
        "autoencoder/times_learning_autoencoders.csv"

    file_times_classification = \
        Dataloader.get_results_directory() + \
        "comparison/times_classification.csv"

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
            my_net.load_state_dict(torch.load(autoencoder_learned_file, map_location=torch.device('cpu')))
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
        device = torch.device("cuda:0")
        print("Checking for multiple GPUs")
        if torch.cuda.device_count() > 1:
            print("Using: \t\t", torch.cuda.device_count(), " GPUs")
            my_net = nn.DataParallel(my_net)
        else:
            print("No multiple GPUs")
        my_net.to(device)

        from torch.autograd import Variable
        num_epochs = 10
        batch_size = 128
        learning_rate = 1e-3
        epsilon = 0.23
        the_best_loss = sys.maxsize
        for epoch in range(num_epochs):
            time_beg = current_time_ms()
            for i, data in enumerate(my_dataloader):
                img, _ = data
                img = img.to(device)
                # img = Variable(img).cpu()
                # ===================forward=====================
                output = my_net(img)
                loss = criterion(output, img)
                # ===================backward====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            # ===================log========================
            time_fin = current_time_ms()
            print('epoch [', epoch + 1, '/', num_epochs, '], loss:', loss.item())
            print('Begin time: ', time_beg)
            print('Begin time: ', time_fin)
            duration = int(time_fin - time_beg)
            print("Duration:  ", duration, " seconds")

            # if loss.item() < epsilon:
            #     print('Epsilon break. Epsilon value: ',epsilon)
            #     break
            save_csv_file(  # saving learning time and loss value
                file_times_learning,
                [Dataloader.get_name(),
                my_net.getType(),
                my_net.getName(),
                epoch + 1,
                dataloader_suffix,
                str(duration),
                loss.item()])
            if loss.item() < the_best_loss:
                the_best_loss = loss.item()
                save_model(
                    my_net,
                    autoencoder_learned_file,
                    autoencoder_learned_file_description,
                    the_best_loss,
                    Dataloader.get_name())


    print()
    print("***   If overwrite data   ***")
    print("---------------------------------")
    print("Overwrite: ", overwrite)
    suffix = ""
    if middle_cut_out:
        suffix = "_mcu"
    param_txt = ""
    if param:
        param_txt = param + "_"

    img_name = \
        Dataloader.get_name() + "_" + \
        Classifier.get_name() + "_" + param_txt + \
        my_net.getType() + '_autoencoder_' + \
        my_net.getName() + suffix + '.png'
    result_img_path = Dataloader.get_results_directory() + 'img/' + img_name
    data_name = \
        Dataloader.get_name() + "_" + \
        Classifier.get_name() + "_" + param_txt + \
        my_net.getType() + '_autoencoder_' + \
        my_net.getName() + suffix + '.txt'
    result_data_path = Dataloader.get_results_directory() + 'data/' + data_name
    print("Result file names:")
    print(result_img_path)
    print(result_data_path)
    if not overwrite and \
        os.path.isfile(result_img_path) and \
        os.path.isfile(result_data_path):
        print("Result files exist ")
        print("Ending procedure [*]")
        return

    print()
    print("***   Autoencoding immage   ***")
    print("---------------------------------")
    print("Image shape: ", the_image_shape)
    print("Image list shape: ", np.shape(the_image_list))
    print("Image code got from autoencoder")
    file_times_autoencoding = \
        Dataloader.get_results_directory() + \
        "autoencoder/times_autoencoding.csv"
    my_net.to(torch.device("cpu"))
    time_beg = current_time_ms()
    the_image_autoencoded = [my_net.getCode(torch.Tensor(point)).detach().numpy() for point in the_image_list]
    time_fin = current_time_ms()
    duration = int(time_fin - time_beg)
    print("Duration:  ", duration, " seconds")
    save_csv_file(  # saving autoencoding time
        file_times_autoencoding, # TODO
        [Dataloader.get_name(),
         my_net.getType(),
         my_net.getName(),
         dataloader_suffix,
         str(duration)])

    print("Autoencoded image shape: ", np.shape(the_image_autoencoded))
    if middle_cut_out:
        print()
        print("***   Middle cut out   ***")
        print("---------------------------------")
        the_image_autoencoded = cut_out_in_midle(the_image_autoencoded, Dataloader)

    print()
    print("***   Clustering   ***")
    print("---------------------------------")
    time_beg = current_time_ms()
    the_image_classified =\
        Classifier.clustering(the_image_autoencoded, the_image_shape, nr_of_clusters, extra_parameters=param)
    time_fin = current_time_ms()
    print('Begin time: ', time_beg)
    print('Finish time: ', time_fin)
    duration = int(time_fin - time_beg)
    print("Duration:  ", duration, " seconds")
    save_csv_file(  # saving classification time
        file_times_classification,
        [Dataloader.get_name(),
         dataloader_suffix,
         my_net.getType(),
         my_net.getName(),
         suffix,
         Classifier.get_name(),
         param_txt,
         duration])
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
        print("Path: ", result_img_path)
        plt.savefig(result_img_path, bbox_inches='tight')

    print()
    print("***   Saving data   ***")
    print("---------------------------------")
    if save_data:
        print("Path: ", result_data_path)
        np.savetxt(result_data_path, the_image_classified, delimiter=" ", newline="\n", header=data_name, fmt="%s")


def run_machine_for_all():
    print()
    print("***   Run machine for all start   ***")
    print("=====================================")
    print()

    autoencoders = []
    autoencoders.append(Autoencoder0)
    autoencoders.append(Autoencoder1)
    autoencoders.append(Autoencoder2)
    autoencoders.append(Autoencoder3)
    autoencoders.append(Autoencoder4)
    autoencoders.append(Autoencoder5)
    autoencoders.append(Autoencoder6)
    autoencoders.append(Autoencoder7)
    autoencoders.append(Autoencoder8)
    autoencoders.append(Autoencoder9)

    dataloaders = []
    dataloaders.append(Dataloader1)
    dataloaders.append(Dataloader2)
    dataloaders.append(Dataloader3)
    dataloaders.append(Dataloader4)
    dataloaders.append(Dataloader5)
    dataloaders.append(Dataloader6)

    clustering_methods = []
    clustering_methods.append(classifier1)
    clustering_methods.append(classifier2)

    for Dataloader in dataloaders:
        for Autoencoder in autoencoders:
            for clustering in clustering_methods:
                try:
                    if clustering == classifier2:
                        for shape in classifier2.cv_types:
                            # run_machine(Autoencoder, Dataloader(), clustering, Dataloader.get_number_of_clusters()
                            #             , first=False, middle_cut_out=True, param=shape)
                            run_machine(Autoencoder, Dataloader(), clustering, Dataloader.get_number_of_clusters()
                                        , first=False, middle_cut_out=False, param=shape)
                    else:
                        # run_machine(Autoencoder, Dataloader(), clustering, Dataloader.get_number_of_clusters()
                        #             , first=False, middle_cut_out=True)
                        run_machine(Autoencoder, Dataloader(), clustering, Dataloader.get_number_of_clusters()
                                    , first=False, middle_cut_out=False)
                    plt.close("all")
                except ValueError:
                    print("EXCEPTION!!!!! -- ValueError")
                    print()
                except Exception:
                    import traceback
                    print("EXCEPTION!!!!!")
                    traceback.print_exc()
                    print()
                    import sys
                    sys.exit()
    print()
    print("***   Run machine for all end   ***")
    print("=====================================")
    print()


def get_timestamp():
    import time
    from datetime import datetime
    date_time_obj = datetime.now()
    date_obj = date_time_obj.date()
    time_obj = date_time_obj.time()
    timestamp_str = str(date_obj.day) + '-' + str(date_obj.month) + '-' + str(date_obj.year) + '_' + \
        str(time_obj.hour) + '^' + str(time_obj.minute) + '^' + str(time_obj.second)
    return timestamp_str


if __name__ == '__main__':
    to_file = True # TODO change

    # Przekierowanie wyjścia do pliku
    if to_file:
        import sys
        orig_stdout = sys.stdout
        output_file = open('results/creator_labeled_image_output_' + get_timestamp() + '.txt', 'w')
        sys.stdout = output_file

    print("START")
    start_time = time.time()
    print("Start time:  ", time.ctime(start_time))

    # Creating result files (if doesnt exists)
    crf.create_result_files()

    # Procedures
    run_machine_for_all()

    # run_machine(
    #      Autoencoder5, Dataloader5(), classifier1, Dataloader5.get_number_of_clusters(),
    #      first=False, middle_cut_out=False, param='tied')

    # end procedures

    print("\nEND")
    end_time = time.time()
    print("End time:  ", time.ctime(end_time))
    print("Duration:  ", int(end_time - start_time), " seconds")

    # Closing file
    if to_file:
        sys.stdout = orig_stdout
        output_file.close()

    # plt.show()
