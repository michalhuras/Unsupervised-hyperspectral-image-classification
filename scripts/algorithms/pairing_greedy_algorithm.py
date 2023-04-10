#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np


class NotEnoughLabelsError(Exception):
    # Raised when nan value is being found in labels distance array
    pass


class PairingAlgorithm:
    def __init__(self):
        self.algorithm_name = "Greedy algorithm"

        self.verbal = True
        self.prefix = "\t"

    def my_print(self, text="", extra_prefix=""):
        if self.verbal:
            print(self.prefix + extra_prefix + str(text))

    @staticmethod
    def my_print_array(array, verbal=True):
        if verbal:
            np.set_printoptions(precision=5)
            print("      ", end="")
            for x in range(array.shape[1]):
                if x < 10:
                    print(" ", end="")
                print(" ", x, "  ", end="  ")
            print()

            for y in range(array.shape[0]):
                if y < 10:
                    print(" ", end="")
                print(y, "  ", end="")
                for x in range(array.shape[1]):
                    print("%0.5f" % array[y][x], end="  ")
                print()

    @staticmethod
    def count_difference(array_1, array_2, show_img=False):
        from algorithms.dif_mean_squared_error import count_difference
        return count_difference(array_1, array_2, show_img=show_img)

    @staticmethod
    def normalise_results(array):
        # Podział wszystkich wartości przez największą wartość bezwzględną
        max_value = np.amax(array)
        return array / max_value

    @staticmethod
    def get_original_index(rows_deleted, columns_deleted, minimal_value_index):
        result = np.copy(minimal_value_index)
        a_rows_deleted = np.sort(rows_deleted)
        for row in a_rows_deleted:
            if row <= result[0]:
                result[0] += 1

        a_columns_deleted = np.sort(columns_deleted)
        for column in a_columns_deleted:
            if column <= result[1]:
                result[1] += 1
        return result

    def math_in_pairs(self, nr_of_labels, ideal_spectral_curve, spectral_curve, verbal=False, plot=False, prefix="\t"):
        self.verbal = verbal
        self.prefix = prefix

        if nr_of_labels is not ideal_spectral_curve.shape[0]:
            ideal_spectral_curve = ideal_spectral_curve.transpose()
        if nr_of_labels is not spectral_curve.shape[0]:
            spectral_curve = spectral_curve.transpose()

        self.my_print()
        self.my_print("Pairing")
        self.my_print("Algorithm name: \t\t\t" + self.algorithm_name)
        distance_array = np.full((nr_of_labels, nr_of_labels), np.float64(-1.0))
        self.my_print(distance_array)

        if plot:
            plt.clf()
            plt.subplot(1, 2, 1)
            plt.plot(np.copy(spectral_curve).transpose())
            plt.title('Labeled image')
            plt.axis('tight')
            plt.subplot(1, 2, 2)
            plt.plot(np.copy(ideal_spectral_curve).transpose())
            plt.title('Ground truth')
            plt.axis('tight')
            plt.show()

        self.my_print()
        self.my_print("Counting differences between labels")
        for y in range(nr_of_labels):
            for x in range(nr_of_labels):
                distance_array[y][x] = self.count_difference(ideal_spectral_curve[x], spectral_curve[y])
        self.my_print_array(distance_array)

        self.my_print()
        self.my_print("Checking if any argument is nan - situation when at least one label wasn't used")
        if np.isnan(distance_array).any():
            print("nan argument in array throwing message!")
            raise NotEnoughLabelsError

        self.my_print()
        self.my_print("Normalization")
        distance_array = self.normalise_results(distance_array)
        self.my_print_array(distance_array, verbal=True)

        self.my_print()
        self.my_print("Finding minimal value and filing result vector")
        pairing_result = np.zeros(nr_of_labels)
        distance_array_cp = np.copy(distance_array)
        distance_sum = 0
        columns_deleted = []
        rows_deleted = []
        for n in range(nr_of_labels):
            minimal_value = np.amin(distance_array_cp)
            minimal_value_index = np.where(distance_array_cp == np.amin(distance_array_cp))
            minimal_value_index = [minimal_value_index[0][0], minimal_value_index[1][0]]
            minimal_value_index_original = self.get_original_index(rows_deleted, columns_deleted, minimal_value_index)
            distance_array_cp = np.delete(distance_array_cp, minimal_value_index[0], 0)
            distance_array_cp = np.delete(distance_array_cp, minimal_value_index[1], 1)
            columns_deleted.append(minimal_value_index_original[1])
            rows_deleted.append(minimal_value_index_original[0])
            distance_sum += minimal_value

            self.my_print()
            self.my_print("After change")
            self.my_print_array(distance_array_cp)
            # self.my_print(distance_array_cp)
            self.my_print("Iteration: \t\t\t" + str(n), extra_prefix="\t\t")
            self.my_print("Minimal value: \t\t" + str(minimal_value), extra_prefix="\t\t")
            self.my_print(
                "Minimal value index: \t\t" + str(minimal_value_index[0]) + "\t" + str(minimal_value_index[1]),
                extra_prefix="\t\t")
            self.my_print(
                "Minimal value index in original image: \t\t"
                + str(minimal_value_index_original[0]) + "\t" + str(minimal_value_index_original[1]),
                extra_prefix="\t\t")
            pairing_result[minimal_value_index_original[0]] = minimal_value_index_original[1]

        self.my_print()
        self.my_print("Pairing result: \t\t\t" + str(pairing_result))
        self.my_print("Distance sum value: \t\t" + str(distance_sum))

        return pairing_result, distance_sum
