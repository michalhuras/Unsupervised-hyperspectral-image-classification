#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


# Rzutowanie wartości na z uint16 na uint8
# https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
def numpy_to_uint8(the_image):
    original_image_type = the_image.dtype
    original_image_info = np.iinfo(original_image_type)
    converted_image = the_image.astype(np.float32) / original_image_info.max
    converted_image_type = np.uint8
    converted_image_info = np.iinfo(converted_image_type)
    converted_image = converted_image_info.max * converted_image
    converted_image = converted_image.astype(np.uint8)
    print("Original image type: ", original_image_type)
    print("Converted image type:  ", converted_image_type)
    return converted_image
