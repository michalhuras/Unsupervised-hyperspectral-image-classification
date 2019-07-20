#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class PairingAlgorithm:
    def __init__(self):
        self.algorithm_name = "Greedy algorithm"

        self.verbal = True
        self.prefix = "\t"

    def my_print(self, text="", extra_prefix=""):
        if self.verbal:
            print(self.prefix + extra_prefix + str(text))

    @staticmethod
    def count_difference(array_1, array_2):
        # Mean squared error
        # https://www.geeksforgeeks.org/python-mean-squared-error/
        return np.square(np.subtract(array_1, array_2)).mean()

    @staticmethod
    def normalise_results(array):
        # Podział wszystkich wartości przez największą wartość bezwzględną
        max_value = np.amax(array)
        return array / max_value

    def math_in_pairs(self, nr_of_labels, ideal_spectral_curve, spectral_curve, verbal=True, prefix="\t"):
        self.verbal = verbal
        self.prefix = prefix

        if nr_of_labels is not ideal_spectral_curve.shape[0]:
            ideal_spectral_curve = ideal_spectral_curve.transpose()
        if nr_of_labels is not spectral_curve.shape[0]:
            spectral_curve = spectral_curve.transpose()

        self.my_print()
        self.my_print("Pairing")
        self.my_print("Algorithm name: \t\t\t" + self.algorithm_name)
        distance_array = np.full((nr_of_labels, nr_of_labels), -1)
        self.my_print(distance_array)

        self.my_print()
        self.my_print("Counting differences between labels")
        for y in range(nr_of_labels):
            for x in range(nr_of_labels):
                distance_array[y][x] = self.count_difference(ideal_spectral_curve[x], spectral_curve[y])
        self.my_print(distance_array)

        self.my_print()
        self.my_print("Normalization")
        distance_array = self.normalise_results(distance_array)
        self.my_print(distance_array)

        self.my_print()
        self.my_print("Finding minimal value and filing result vector")
        pairing_result = np.zeros(nr_of_labels)
        distance_array_cp = np.copy(distance_array)
        distance_sum = 0
        for n in range(nr_of_labels):
            minimal_value = np.amin(distance_array_cp)
            minimal_value_index = np.where(distance_array_cp == np.amin(distance_array_cp))
            minimal_value_index_original = np.where(distance_array == np.amin(distance_array_cp))
            distance_array_cp = np.delete(distance_array_cp, minimal_value_index[0][0], 0)
            distance_array_cp = np.delete(distance_array_cp, minimal_value_index[1][0], 1)
            distance_sum += minimal_value

            self.my_print()
            self.my_print(distance_array_cp, extra_prefix="\t\t")
            self.my_print("Iteration: \t\t\t" + str(n), extra_prefix="\t\t")
            self.my_print("Minimal value: \t\t" + str(minimal_value), extra_prefix="\t\t")
            self.my_print(
                "Minimal value index: \t\t" + str(minimal_value_index[0][0]) + "\t" + str(minimal_value_index[1][0]),
                extra_prefix="\t\t")
            self.my_print(
                "Minimal value index in original image: \t\t"
                + str(minimal_value_index_original[0][0]) + "\t" + str(minimal_value_index_original[1][0]),
                extra_prefix="\t\t")
            pairing_result[minimal_value_index_original[1][0]] = minimal_value_index_original[0][0]

        self.my_print()
        self.my_print("Pairing result: \t\t\t" + str(pairing_result))
        self.my_print("Distance sum value: \t\t" + str(distance_sum))

        return pairing_result, distance_sum
