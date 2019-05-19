#!/usr/bin/env python
# -*- coding: utf-8 -*-

from spectral import *
from scipy import io
import numpy as np

"""
    Skrypt musi być uruchomiany w interpreterze ipython
    
    Konfiguracja:
        pip install ipython
        + pip install <biblioteki wymienione powyżej>

    Przed uruchomieniem należy: 
        zmienić filename w skrypcie
        
    W wierszu polecenia: 
        ipython 
        cd <katalog>
        run visualisatiob_SPY.py
        
    
    About IPython:
        https://www.codecademy.com/articles/how-to-use-ipython
    SPY documentation: 
        http://www.spectralpython.net/graphics.html

"""

if __name__ == '__main__':
    print("START")
    filename = 'C:/TestingCatalog/AI_data/Indian Pines/Indian_pines_corrected.mat'
    ImDict = io.loadmat(filename)
    the_image = ImDict['indian_pines_corrected']

    image_size = np.shape(the_image)
    NRows = image_size[0]
    NCols = image_size[1]
    NBands = image_size[2]

    # przykład
    # img = open_image('92AV3C.lan')

    print(type(the_image))

    # Rzutowanie wartości na z uint16 na uint8
    # https://stackoverflow.com/questions/46689428/convert-np-array-of-type-float64-to-type-uint8-scaling-values
    original_image_type = the_image.dtype
    original_image_info = np.iinfo(original_image_type)
    converted_image = the_image.astype(np.float32) / original_image_info.max
    converted_image_type = np.uint8
    converted_image_info = np.iinfo(converted_image_type)
    converted_image = converted_image_info.max * converted_image
    converted_image = converted_image.astype(np.uint8)
    print("Original image type: ", original_image_type)
    print("Converted image type:  ", converted_image_type)

    # Wyświetlanie obrazka
    view = imshow(converted_image, (29, 20, 11))
    view = imshow(the_image, (29, 20, 11))
    print(view)

    # Wyświtlanie kostki - nie działa
    # import spectral
    # spectral.settings.WX_GL_DEPTH_SIZE = 16
    # print(np.size(the_image))
    # print(np.shape(the_image))
    # smaller_img = the_image[0:30, 0:30, 0:30]
    # print(np.size(smaller_img))
    # print(np.shape(smaller_img))
    # view_cube(smaller_img, bands=[29, 20, 11])
    print("END")
