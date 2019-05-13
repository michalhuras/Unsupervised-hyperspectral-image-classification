import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import linalg
from scipy import io
from mpl_toolkits.mplot3d import Axes3D

import draw_spectral_curve as draw

''' Rysowanie obrazu RGB na podstawie obrazu spektralnego
    https://www.neonscience.org/classification-pca-python
'''

def PlotSpectraAndMean(Spectra, Wv, fignum):
    # Spectra is NBands x NSamps
    mu = np.mean(Spectra, axis=1)
    print(np.shape(mu))
    plt.figure(fignum)
    plt.plot(Wv, Spectra, 'c')
    plt.plot(Wv, mu, 'r')
    plt.show()
    return mu


datadir = 'C:/TestingCatalog/AI_data/RSDI2017-Data-SpecClass/'

filename = 'C:/TestingCatalog/AI_data/Indian Pines/Indian_pines_corrected.mat'
# filename   = datadir + 'OSBSTinyIm.mat'
# filename   = 'C:/TestingCatalog/AI_data/Indian Pines/Indian_pines.mat'

ImDict = io.loadmat(filename)
OSBSTinyIm = ImDict['indian_pines_corrected']
# OSBSTinyIm = ImDict['OSBSTinyIm']
# OSBSTinyIm = ImDict['indian_pines']

# Size
TinySize = np.shape(OSBSTinyIm)
NRows = TinySize[0]
NCols = TinySize[1]
NBands = TinySize[2]
print('Size(rows, cols, bands): {0:4d} {1:4d} {2:4d}'.format(NRows, NCols, NBands))

# Adding colors
Wv = io.loadmat(datadir + "NEONWvsNBB")
Wv = Wv['NEONWvsNBB']
print(np.shape(Wv))
#plt.figure(1)
#plt.plot(range(346), Wv)
#plt.show()

### HAVE TO SUBTRACT AN OFFSET BECAUSE OF BAD BAND ###
### REMOVAL AND 0-BASED Python vs 1-Based MATLAB   ###
Offset     = 7

### LOAD & PRINT THE INDICES FOR THE COLORS   ###
### AND DIG THEM OUT OF MANY LAYERS OF ARRAYS ###

NEONColors = io.loadmat(datadir + 'NEONColors.mat')
NEONRed    = NEONColors['NEONRed']
NEONGreen  = NEONColors['NEONGreen']
NEONBlue   = NEONColors['NEONBlue']

NEONNir    = NEONColors['NEONNir']
print(NEONNir)

NEONRed    = NEONRed[0][0]-Offset
NEONGreen  = NEONGreen[0][0]-Offset
NEONBlue   = NEONBlue[0][0]-Offset
NEONNir    = NEONNir[0][0]-Offset
print('Indices:     {0:4d} {1:4d} {2:4d} {3:4d}'.format(NEONRed, NEONGreen, NEONBlue, NEONNir))

### CONVERT THE INDICES TO WAVELENGTHS ###
NEONRedWv    = Wv[NEONRed][0]
NEONGreenWv  = Wv[NEONGreen][0]
NEONBlueWv   = Wv[NEONBlue][0]
NEONNirWv    = Wv[NEONNir][0]
print('Wavelengths: {0:4d} {1:4d} {2:4d} {3:4d}'.format(NEONRedWv, NEONGreenWv, NEONBlueWv, NEONNirWv))

# Colour image
#print (NEONRed, NEONGreen, NEONBlue)
#print (type (NEONRed))

#RGBIm = OSBSTinyIm[:, :, [NEONRed, NEOXNGreen, NEONBlue]]
RGBIm = OSBSTinyIm[:, :, [29, 20, 11]]

#RGBIm = np.sqrt(RGBIm)
#print(RGBIm)
print("---------------------")
#print((RGBIm * 255).astype(np.uint8))
#print(type((RGBIm * 255).astype(np.uint8)))

plt.figure(2)
plt.imshow((RGBIm * 255).astype(np.uint8))
plt.show()

print("end")