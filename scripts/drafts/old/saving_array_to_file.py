#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

if __name__ == '__main__':
    output_file_name_1 = "output_1.txt"
    output_file_name_2 = "output_2.txt"

    array_1 = np.array([[1, 2, 3], [4, 5, 6]])
    array_2 = np.array([[1.123456789, 2.123456789], [3.123456789, 4.123456789], [5.123456789, 6.123456789]])

    header_1 = "Array 1"
    header_2 = "Array 2"

    np.savetxt(output_file_name_1, array_1, delimiter=" ", newline="\n", header=header_1, fmt="%s")
    np.savetxt(output_file_name_2, array_2, fmt="%s")

    new_array_1 = np.loadtxt(output_file_name_1, delimiter=" ")
    new_array_2 = np.loadtxt(output_file_name_2, delimiter=" ")

    print(array_1)
    print()
    print(new_array_1)
    print()
    print()

    print(array_2)
    print()
    print(new_array_2)
    print()
