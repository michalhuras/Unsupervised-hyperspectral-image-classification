#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

'''
   Spectral curve dataloader
'''


class Dataloader():
    def __init__(self):
        self.name = 'spectral_curve_dataloader'

    def get_spectral_curve_from_file(self, input_file_path, verbal=True):
        if verbal:
            print()
            print("***   Get spectral curve from file   ***")
            print("---------------------------------")

        spectral_curve = np.loadtxt(input_file_path, delimiter=" ")

        return spectral_curve

if __name__ == '__main__':
    print("SPECTRAL CURVE DATALOADER")
