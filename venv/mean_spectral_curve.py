import torch
import torch.nn as nn
from scipy import io
import numpy as np
import time


def create_spectral_curve(
        dataloader,
        show_img=True,
        save_img=True,
        save_data=True,
        draw_legend=True,
        output_name=""):

    print("\n\n\n\n\n\n")
    print("----------------")
    print("START")
    print("Data name: ", dataloader.get_name())
    image = dataloader.get_image()
    labels = dataloader.get_labels()
    number_of_labels = dataloader.get_number_of_clusters()
    shape = dataloader.get_image_shape()

    elements_in_labels = np.zeros(number_of_labels)
    # zbieram ilość elementów, dla etykiet
    print("Kształ tabeli ilości elementów etykiet: ", np.shape(elements_in_labels))

    sum_of_clusters = np.zeros((number_of_labels, shape[2]))
    # zbieram sumę wartości hiperspektralnych dla każdego piksela
    print("Kształt tabeli zbierającej sume wartości dla etykiet: ", np.shape(sum_of_clusters))

    # import matplotlib.pyplot as plt
    # plt.imshow(labels)
    # plt.show()

    x = 0
    y = 0
    for i in range(shape[0]*shape[1]):
        sum_of_clusters[int(labels[y, x])] += image[y, x]
        elements_in_labels[int(labels[y, x])] += 1
        x = x + 1
        if x == shape[1]:
            x = 0
            y += 1

    # if show_img:
    #     import matplotlib.pyplot as plt
    #     plt.title("Krzywe spektralne sumy poetykietowanych wartości")
    #     for i in range(number_of_labels):
    #         plt.plot(sum_of_clusters[i], label=str(i))
    #     plt.legend()
    #     plt.axis('tight')
    #     plt.show()

    print("Ilość elementów zaetykietowanych: ", elements_in_labels)

    spectral_curve = np.copy(sum_of_clusters)
    for i in range(number_of_labels):
        spectral_curve[i] = sum_of_clusters[i]/elements_in_labels[i]

    if show_img:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.title("Krzywe spektralne")
        for i in range(number_of_labels):
            plt.plot(spectral_curve[i], label=str(i))
        if draw_legend:
            plt.legend()
        plt.axis('tight')
        plt.show()

    if save_img:
        print("+++++++ SAVING IMAGE +++++++")
        import matplotlib.pyplot as plt
        plt.clf()
        plt.title("Krzywe spektralne")
        for i in range(number_of_labels):
            plt.plot(spectral_curve[i], label=str(i))
        if draw_legend:
            plt.legend()
        plt.axis('tight')
        img_name = ""
        if output_name == "":
            img_name = "IDEAL_spectral_curve.png"
        else:
            img_name = "spectral_curve_" + output_name + ".png"
        result_img_path = dataloader.get_results_directory() + 'img/' + img_name
        print("Path: ", result_img_path)
        plt.savefig(result_img_path, bbox_inches='tight')
        # plt.show()

    if save_data:
        print("+++++++ SAVING DATA +++++++")
        data_name = ""
        if output_name == "":
            data_name = "IDEAL_spectral_curve.txt"
        else:
            data_name = "spectral_curve_" + output_name + ".txt"
        result_data_path = dataloader.get_results_directory() + 'data/' + data_name
        print("Path: ", result_data_path)
        np.savetxt(result_data_path, spectral_curve, delimiter=" ", newline="\n", header=data_name, fmt="%s")

    print("END")


if __name__ == '__main__':
    from drafts.tests.test_dataloader import Dataloader as AAA
    create_spectral_curve(AAA())

    from dataloader.indian_pines_dataloader import Dataloader as BBB
    create_spectral_curve(BBB())

    from dataloader.samson_dataloader import Dataloader as CCC
    create_spectral_curve(CCC())

    from dataloader.jasper_ridge_dataloader import Dataloader as DDD
    create_spectral_curve(DDD())

    from dataloader.salinas_dataloader import Dataloader as EEE
    create_spectral_curve(EEE())

    from dataloader.salinas_a_dataloader import Dataloader as FFF
    create_spectral_curve(FFF())

    from dataloader.pavia_dataloader import Dataloader as GGG
    create_spectral_curve(GGG())
