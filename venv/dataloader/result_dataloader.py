#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

'''
   Result dataloader
'''


class Dataloader():
    def __init__(self):
        self.name = 'result_dataloader'

        self.image_shape = ()
        self.nr_of_clusters = 0
        self.image = ()
        self.image_list = ()
        self.image_labels = ()

    def get_image_labels_from_file(self, input_file_path, verbal=False):
        if verbal:
            print()
            print("***   Get labels from result file   ***")
            print("---------------------------------")

        image = np.loadtxt(input_file_path, delimiter=" ")

        return image

if __name__ == '__main__':
    print("RESULT DATALOADER")
