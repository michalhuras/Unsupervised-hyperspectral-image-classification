#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

path = "./data/processed"
datasets_names = ["IndianPines", "JasperRidge", "Pavia", "Salinas", "SalinasA", "Samson"]
result_folders = ["autoencoder", "comparison", "data", "img"]


def create_result_files():
    for name in datasets_names:
        sub_path = path + "/" + name
        if os.path.exists(sub_path):
            print("Path: ", sub_path, "   exists")
        else:
            print("Path: ", sub_path, "   DOESN'T exists")
            print("Creating path: ", sub_path)
            os.mkdir(sub_path)

        for folder in result_folders:
            sub_sub_path = sub_path + "/" + folder
            if os.path.exists(sub_sub_path):
                print("Path: ", sub_sub_path, "   exists")
            else:
                print("Path: ", sub_sub_path, "   DOESN'T exists")
                print("Creating path: ", sub_sub_path)
                os.mkdir(sub_sub_path)


from scripts.dataloader.indian_pines_dataloader import Dataloader as Dataloader1
# from dataloader.indian_pines_cut_out_dataloader import Dataloader as Dataloader11
from scripts.dataloader.jasper_ridge_dataloader import Dataloader as Dataloader2
from scripts.dataloader.pavia_dataloader import Dataloader as Dataloader3
# from dataloader.pavia_cut_out_dataloader import Dataloader as Dataloader33
from scripts.dataloader.salinas_dataloader import Dataloader as Dataloader4
# from dataloader.salinas_cut_out_dataloader import Dataloader as Dataloader44
from scripts.dataloader.salinas_a_dataloader import Dataloader as Dataloader5
# from dataloader.salinas_a_cut_out_dataloader import Dataloader as Dataloader55
from scripts.dataloader.samson_dataloader import Dataloader as Dataloader6
import numpy as np
import matplotlib.pyplot as plt

def create_ground_truth_img():
    dataloaders = []
    dataloaders.append(Dataloader1)
    # dataloaders.append(Dataloader11)
    dataloaders.append(Dataloader2)
    dataloaders.append(Dataloader3)
    # dataloaders.append(Dataloader33)
    dataloaders.append(Dataloader4)
    # dataloaders.append(Dataloader44)
    dataloaders.append(Dataloader5)
    # dataloaders.append(Dataloader55)
    dataloaders.append(Dataloader6)

    for Dataloader in dataloaders:
        dataloader = Dataloader()
        result_img_path = dataloader.get_results_directory() + 'img/GT_' + dataloader.get_name() + '.png'
        ground_truth = dataloader.get_labels(verbal=False)
        print("Result shape: ", np.shape(ground_truth))
        plt.imshow(ground_truth)
        plt.show()
        plt.imshow(ground_truth)
        print("Path: ", result_img_path)
        plt.savefig(result_img_path, bbox_inches='tight')


if __name__ == '__main__':
    create_result_files()
    create_ground_truth_img()
