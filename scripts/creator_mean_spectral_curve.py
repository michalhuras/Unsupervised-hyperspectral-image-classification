import numpy as np

# from drafts.tests.test_dataloader import Dataloader as test_dataloader

from scripts.dataloader.indian_pines_dataloader import Dataloader as indian_pines_dataloader
from scripts.dataloader.samson_dataloader import Dataloader as samson_dataloader
from scripts.dataloader.jasper_ridge_dataloader import Dataloader as jasper_ridge_dataloader
from scripts.dataloader.salinas_dataloader import Dataloader as salinas_dataloader
from scripts.dataloader.salinas_a_dataloader import Dataloader as salinas_a_dataloader
from scripts.dataloader.pavia_dataloader import Dataloader as pavia_dataloader


def create_spectral_curve(
        dataname,
        image,
        labels,
        number_of_labels,
        shape,
        results_directory,
        show_img=True,
        save_img=True,
        save_data=True,
        draw_legend=True,
        output_name="",
        zero_out=False):
    # Note: zero_out - when creating spectral curves for ground truth labeling this should be set to true
    #       When true spectral curve doesn't contain zero label(background)
    print("---------------------------------------------")
    print("START")
    print("Data name: ", dataname)

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
        element = int(labels[y, x])
        if zero_out:
            if element != 0:
                element = element - 1
                # klastry zostaną zapisane w tabeli od 0, dlatego 1 musi się zmienić na 0
                sum_of_clusters[element] += image[y, x]
                elements_in_labels[element] += 1
        else:
            sum_of_clusters[element] += image[y, x]
            elements_in_labels[element] += 1

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
        result_img_path = results_directory + 'img/' + img_name
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
        result_data_path = results_directory + 'data/' + data_name
        print("Path: ", result_data_path)
        np.savetxt(result_data_path, spectral_curve, delimiter=" ", newline="\n", header=data_name, fmt="%s")

    print("END")
    return spectral_curve


def create_spectral_curve_from_dataloader(
        dataloader,
        show_img=True,
        save_img=True,
        save_data=True,
        draw_legend=True,
        output_name=""):

    print("\ncreate_spectral_curve_from_dataloader")

    return create_spectral_curve(
        dataloader.get_name(verbal=False),
        dataloader.get_image(verbal=False),
        dataloader.get_labels(verbal=False),
        dataloader.get_number_of_clusters(verbal=False),
        dataloader.get_image_shape(verbal=False),
        dataloader.get_results_directory(verbal=False),
        show_img=False,
        save_img=True,
        save_data=True,
        draw_legend=True,
        output_name="",
        zero_out=True)


def create_spectral_curve_from_dataloader_plus(
        dataloader,
        labels,
        show_img=True,
        save_img=True,
        save_data=True,
        draw_legend=True,
        output_name=""):

    print("\ncreate_spectral_curve_from_dataloader")

    return create_spectral_curve(
        dataloader.get_name(verbal=False),
        dataloader.get_image(verbal=False),
        labels,
        dataloader.get_number_of_clusters(verbal=False),
        dataloader.get_image_shape(verbal=False),
        dataloader.get_results_directory(verbal=False),
        show_img=show_img,
        save_img=save_img,
        save_data=save_data,
        draw_legend=draw_legend,
        output_name=output_name)


def create_spectral_curve_for_ideal_data():
    # create_spectral_curve_from_dataloader(test_dataloader())

    create_spectral_curve_from_dataloader(indian_pines_dataloader())
    create_spectral_curve_from_dataloader(samson_dataloader())
    create_spectral_curve_from_dataloader(jasper_ridge_dataloader())
    create_spectral_curve_from_dataloader(salinas_dataloader())
    create_spectral_curve_from_dataloader(salinas_a_dataloader())
    create_spectral_curve_from_dataloader(pavia_dataloader())


def create_spectral_curve_for_existing_data():
    import os

    print("Searching for result files")
    result_directories_with_dataloaders = {
        "./data/processed/IndianPines/data/": indian_pines_dataloader(),
        "./data/processed/JasperRidge/data/": jasper_ridge_dataloader(),
        "./data/processed/Pavia/data/": pavia_dataloader(),
        "./data/processed/Salinas/data/": salinas_dataloader(),
        "./data/processed/SalinasA/data/": salinas_a_dataloader(),
        "./data/processed/Samson/data/": samson_dataloader(),
        # "./data/processed/tests/data":test_dataloader(),
    }

    # name: directory
    for path in result_directories_with_dataloaders:
        names_and_directories = {}
        print("\tPath: ", path)
        print("\tDataloader name: ", result_directories_with_dataloaders[path].get_name(False))

        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if '.txt' in file \
                        and "spectral_curve" not in file\
                        and "report_" not in file:
                    names_and_directories[file] = os.path.join(r, file)

        print(names_and_directories)

        for file_name in names_and_directories:
            from scripts.dataloader.result_dataloader import Dataloader as ResoultDataloader
            dataloader = ResoultDataloader()
            image_labels =\
                dataloader.get_image_labels_from_file(names_and_directories[file_name], verbal=False)

            # import matplotlib.pyplot as plt
            # plt.imshow(image)
            # plt.show()

            if file_name.endswith('.txt'):
                file_name = file_name[:-4]
            print("\t create_spectral_curve_from_dataloader_plus:  ", file_name)
            print("\t Dir: ", result_directories_with_dataloaders[path].get_name(False))
            create_spectral_curve_from_dataloader_plus(
                result_directories_with_dataloaders[path],
                image_labels,
                output_name=file_name,
                show_img=False)


if __name__ == '__main__':
    # Ideal data
    create_spectral_curve_for_ideal_data()

    # All data
    create_spectral_curve_for_existing_data()

    # Available results files and dataloaders:
    # "./data/processed/IndianPines/data/"     indian_pines_dataloader()
    # "./data/processed/JasperRidge/data/"     jasper_ridge_dataloader()
    # "./data/processed/Pavia/data/"           pavia_dataloader()
    # "./data/processed/Salinas/data/"         salinas_dataloader()
    # "./data/processed/SalinasA/data/"        salinas_a_dataloader()
    # "./data/processed/Samson/data/"          samson_dataloader()
    # # "./result/tests/data"           test_dataloader()

    '''
    # Example use for one result file
    from dataloader.result_dataloader import Dataloader as ResoultDataloader
    dataloader = ResoultDataloader()
    file_directory = "./data/processed/Samson/data/clustering_kmeans_linear_autoencoder_1.txt"
    image_labels = \
        dataloader.get_image_labels_from_file(file_directory, verbal=False)

    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()

    file_name = "clustering_kmeans_linear_autoencoder_1"
    create_spectral_curve_from_dataloader_plus(
        samson_dataloader(),
        image_labels,
        output_name=file_name,
        show_img=True)
    '''

