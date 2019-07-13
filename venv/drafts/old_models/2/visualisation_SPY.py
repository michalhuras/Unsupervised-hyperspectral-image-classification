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

data_dir = 'C:/Users/Public/AI/artificial-intelligence---my-beginning/venv/data/Samson/'

if __name__ == '__main__':
    print("START")

    print()
    print("***   Loading data   ***")
    print("---------------------------------")
    filename = 'samson_1.mat'
    ImDict = io.loadmat(data_dir + filename)
    image_name = 'V'
    the_image = ImDict[image_name]
    the_image = the_image.transpose()
    image_size = np.shape(the_image)
    print(image_size)
    NRows = image_size[0]
    NCols = image_size[1]
    # NBands = image_size[2]
    print("Lokalizacja obrazu: \t", data_dir + filename)
    print("Nazwa obrazu:  \t\t\t", image_name)
    print("Rozmiar: \t\t\t\t", "wiersze: ", NRows, " kolumny: ", NCols) #, " zakresy: ", NBands)
    print("Ilośc pikseli (ilość kolumn * ilość wierszy): ", NRows * NCols)

    the_image = np.reshape(the_image, (95, 95, 156))

    # przykład
    # img = open_image('92AV3C.lan')

    print(type(the_image))
    # converted_image = mo.numpy_to_uint8(the_image)

    # Wyświetlanie obrazka
    # view = imshow(converted_image, (29, 20, 11))
    view = imshow(the_image, (30, 15, 9))
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
