#!/usr/bin/env python
# -*- coding: utf-8 -*-

from spectral import *
from scipy import io
import numpy as np
import mathematical_operations as mo

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
    filename = '/home/michalhuras/Pulpit/artificial-intelligence---my' \
               '-beginning-master/Indian_pines_corrected.mat'
    # filename = 'C:/TestingCatalog/AI_data/Indian Pines/Indian_pines_corrected.mat'
    ImDict = io.loadmat(filename)
    the_image = ImDict['indian_pines_corrected']

    image_size = np.shape(the_image)
    NRows = image_size[0]
    NCols = image_size[1]
    NBands = image_size[2]

    # przykład
    # img = open_image('92AV3C.lan')

    print(type(the_image))
    converted_image = mo.numpy_to_uint8(the_image)

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
