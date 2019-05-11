import matplotlib.pyplot as plt
import numpy as np

# input numpy.ndarray - spectral image, int - x point, int - y point
def draw_spectral_curve(image, x, y):
    if image.ndim != 3 or image.shape[0] < x or image.shape[1] < y:
        print("ERROR !!!")
        print("Dimension error in function: " + __name__)
        print(str(image.shape) + "  " + str(x) + "  "  + str(y))
        return -1
    points = image[x, y, :]
    number_of_layers = image.shape[2]
    sqrt_points = np.sqrt(points)
    #plt.figure(1)
    plt.plot(range(number_of_layers), sqrt_points, label= ("x: "+ str(x) + " ,y: " + str(y)))
    #plt.show()
    return 1

def show():
    plt.legend()
    plt.show()

"""
Example use:
    draw.draw_spectral_curve(image, 3, 3)
    draw.draw_spectral_curve(image, 4, 4)
    draw.draw_spectral_curve(image, 5, 5)
    draw.draw_spectral_curve(image, 6, 6)
    draw.draw_spectral_curve(image, 500, 500)
    draw.show()
"""