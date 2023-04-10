#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# Rzutowanie wartości na z uint16 na uint8
# https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
def numpy_to_uint8(the_image, verbal=False):
    original_image_type = the_image.dtype
    original_image_info = np.iinfo(original_image_type)
    converted_image = the_image.astype(np.float32) / original_image_info.max
    converted_image_type = np.uint8
    converted_image_info = np.iinfo(converted_image_type)
    converted_image = converted_image_info.max * converted_image
    converted_image = converted_image.astype(np.uint8)
    if verbal:
        print("Original image type: ", original_image_type)
        print("Converted image type:  ", converted_image_type)
    return converted_image


def turn_image_in_list(the_image_list, the_image_shape):
    x = 0
    y = 0
    the_image = np.zeros((the_image_shape[0], the_image_shape[1], the_image_shape[2]))
    for i in range(the_image_shape[0] * the_image_shape[1]):
        the_image[x][y] = the_image_list[i] # specjalnie [x][y] zamiast [y][x] !!!
        x = x + 1
        if x == the_image_shape[1]:
            x = 0
            y = y + 1

    the_image_turned = np.reshape(the_image, (the_image_shape[0] * the_image_shape[1], the_image_shape[2]))

    return the_image_turned
