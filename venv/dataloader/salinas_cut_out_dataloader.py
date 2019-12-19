# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as utils
# import torchvision.transforms as transforms
# from scipy import io
# import numpy as np
# import scripts_my.mathematical_operations as mo
# import time
#
# try:
#     from dataloader.salinas_dataloader import Dataloader as BasicDataloader
# except:
#     from salinas_dataloader import Dataloader as BasicDataloader
#
# '''
#     Salinas cut out
# '''
#
#
# class Dataloader(BasicDataloader):
#     def __init__(self):
#         BasicDataloader.__init__(self)
#         self.name = 'salinas_cut_out'
#
#         self.image = ()
#         self.image_list = ()
#         self.image_labels = ()
#
#         self.image_exists = False
#         self.image_list_exists = False
#         self.image_labels_exists = False
#
#         self.background_label = 0
#
#     # def get_name(self, verbal=true):
#     # def get_results_directory(self, verbal=true):
#     # def get_number_of_clusters(verbal=true):
#
#     def get_image(self, verbal=False):
#         if verbal:
#             print()
#             print("***   Get image (cut out)   ***")
#             print("---------------------------------")
#
#         if not self.image_exists:
#             the_image = BasicDataloader.get_image(self, verbal=verbal)
#             the_labels = BasicDataloader.get_labels(self, verbal=verbal)
#
#             if verbal:
#                 print("Lokalizacja obrazu: \t", self.data_dir)
#                 print(
#                     "Rozmiar: \t\t\t\t", "wiersze: ", the_image.shape[0],
#                     " kolumny: ", the_image.shape[1],
#                     " głębokość: ", the_image.shape[2])
#                 print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", the_image.shape[0] * the_image.shape[1])
#
#             if verbal:
#                 print()
#                 print("***   Cutting out background   ***")
#                 print("---------------------------------")
#
#             for row in range(the_image.shape[0]):
#                 for column in range(the_image.shape[1]):
#                     if the_labels[row][column] == self.background_label:
#                         the_image[row][column] = np.zeros((the_image.shape[2]))
#
#             self.image = the_image
#             self.image_exists = True
#
#         return self.image
#
#     def get_image_list(self, verbal=False):
#         if verbal:
#             print()
#             print("***   Get image list (cut out)  ***")
#             print("---------------------------------")
#
#         if not self.image_list_exists:
#             the_image = self.get_image(False)
#             the_image_list = np.reshape(the_image, (self.image_shape[0] * self.image_shape[1], self.image_shape[2]))
#
#             image_size = np.shape(the_image_list)
#             length = image_size[0]
#             depth = image_size[1]
#             if verbal:
#                 print("Rozmiar: \t\t\t\t", "długość: ", length, " głębokość: ", depth)
#                 print("Ilośc pikseli (długość * głębokość): ", length * depth)
#             self.image_list = the_image_list
#             self.image_list_exists = True
#
#         return self.image_list
#
#     # def get_image_shape(self, verbal=true):
#     # def get_labels(self, verbal=true):
#     # def get_dataloader(self, verbal=true):
#
#
# if __name__ == '__main__':
#     my_dataloader = Dataloader()
#     print("\nTEST GET NAME")
#     print("RESULT:  ", my_dataloader.get_name())
#     print("\nTEST GET results directory")
#     print("RESULT:  ", my_dataloader.get_results_directory())
#     print("\nTEST GET NUMBER OF CLUSTERS")
#     print("RESULT:  ", my_dataloader.get_number_of_clusters())
#     print("\nTEST GET IMAGE")
#     print("RESULT:  ", np.shape(my_dataloader.get_image()))
#     print("\nTEST GET IMAGE LIST")
#     print("RESULT:  ", np.shape(my_dataloader.get_image_list()))
#     print("\nTEST GET IMAGE SHAPE")
#     print("RESULT:  ", my_dataloader.get_image_shape())
#     print("\nTEST GET LABELS")
#     print("RESULT:  ", np.shape(my_dataloader.get_labels()))
#     print("\nTEST GET DATALOADER")
#     print("RESULT:  ", my_dataloader.get_dataloader())
