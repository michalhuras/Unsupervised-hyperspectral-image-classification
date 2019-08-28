#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

path = "./results"
datasets_names = ["IndianPines", "JasperRidge", "Pavia", "Salinas", "SalinasA", "Samson"]
result_folders = ["autoencoder", "comparison", "data", "img"]


def create_result_files():
    if os.path.exists("./results"):
        print("Path: ", path, "   exists")
    else:
        print("Path: ", path, "   DOESN'T exists")
        print("Creating path: ", path)
        os.mkdir(path)

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


if __name__ == '__main__':
    create_result_files()
