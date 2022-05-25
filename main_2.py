from skimage.io import imread
from skimage.filters import frangi
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
import imageio
from skimage.morphology import (erosion, dilation, opening,
                                area_closing, area_opening)


def create_comparison(prediction, mask):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    x_dim, y_dim = prediction.shape
    output = []
    for j in range(x_dim):
        row = []
        for k in range(y_dim):

            if prediction[j][k] == 0 and mask[j][k] == 0:
                row.append((0, 0, 0))
                TN += 1
            elif prediction[j][k] == 255 and mask[j][k] == 0:
                row.append((0, 255, 0))
                FP += 1
            elif prediction[j][k] == 0 and mask[j][k] == 255:
                row.append((255, 0, 0))
                FN += 1
            else:
                row.append((255, 255, 255))
                TP += 1
        output.append(row)
    return (TP, FP, TN, FN), output


def calculate_error_measures(data):
    TP, FP, TN, FN = data
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)

    return accuracy, sensitivity, specificity


def create_bitmap(array, threshold):
    x_dim, y_dim = array.shape
    for j in range(x_dim):
        for k in range(y_dim):
            if array[j][k] >= threshold:
                array[j][k] = 255
            else:
                array[j][k] = 0
    return array


def normalize_dataset(dataset, set_max):
    # pobranie wymiarów tablicy
    dataset_height = len(dataset)
    dataset_width = len(dataset[0])

    max_value = 0
    for x in range(dataset_height):
        for y in range(dataset_width):
            if dataset[x][y] > max_value:
                max_value = dataset[x][y]

    if max_value == 0:
        print("normalize_dataset failure, max_value=0")
        return None

    # obliczenie mnożnika
    factor = set_max / max_value

    new_dataset = dataset

    # przemnożenie wszystkich elementów
    for x in range(dataset_height):
        for y in range(dataset_width):
            new_dataset[x][y] = int(dataset[x][y] * factor)

    return new_dataset


def save_img(filename, array):
    numpy_array = np.array(array)
    imageio.imwrite("output/" + filename, numpy_array.astype(np.uint8))
    return


def multi_dil(im, num, element):
    for i in range(num):
        im = dilation(im, element)
    return im


def multi_ero(im, num, element):
    for i in range(num):
        im = erosion(im, element)
    return im


def preprocessing(array):
    array_green = array[:, :, 1]

    kernel = np.array([[-1, -2, -1], [-2, 13, -2], [-1, -2, -1]])
    array_sharpened = cv2.filter2D(src=array_green, ddepth=-1, kernel=kernel)

    array_filter = create_bitmap(normalize_dataset(frangi(array_sharpened), 255), 3)

    element = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])

    array_morph = opening(array_filter)
    array_morph = multi_dil(array_morph, 1, element)
    array_morph = area_opening(area_closing(array_morph, 1000), 1000)

    return array_morph


def postprocessing(array):
    element = np.array([[0, 1, 0],
                        [1, 1, 1],
                        [0, 1, 0]])

    array = opening(array)
    array = multi_dil(array, 1, element)
    array = area_opening(area_closing(array, 1000), 1000)
    return array


def print_quality(data):
    TP, FP, TN, FN = data

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    precision = TP / (TP + FP)
    f_measure = (2 * precision * sensitivity) / (precision + sensitivity)

    print("TP: ", TP, ", ",
          "FP: ", FP, ", ",
          "TN: ", TN, ", ",
          "FN: ", FN)
    print("accuracy: ", accuracy)
    print("sensitivity: ", sensitivity)
    print("specificity: ", specificity)
    print("precision: ", precision)
    print("f_measure: ", f_measure)


def main():
    image_mask_array = []
    image_source_array = []
    image_fov_array = []

    for i in range(15):
        if i+1 < 10:
            number = "0" + str(i + 1)
        else:
            number = str(i + 1)

        image_mask = imread("data/healthy_manualsegm/" + number + "_h.tif")
        image_source = imread("data/healthy/" + number + "_h.jpg")
        image_fov = imread("data/healthy_fovmask/" + number + "_h_mask.tif")

        image_source_preprocessed = preprocessing(image_source)

        image_mask_array.append(image_mask)
        image_source_array.append(image_source_preprocessed)
        image_fov_array.append(image_fov)

    classifier = KNeighborsClassifier(
        n_neighbors=1,
        weights='distance',
        algorithm='kd_tree',
        leaf_size=30,
        p=2,
        n_jobs=-1
    )

    for i in range(11, 12):
        classifier.fit(
            image_source_array[i],
            image_mask_array[i]
        )

    image_prediction_12 = postprocessing(classifier.predict(
        image_source_array[12]
    ))
    image_prediction_13 = postprocessing(classifier.predict(
        image_source_array[13]
    ))
    image_prediction_14 = postprocessing(classifier.predict(
        image_source_array[14]
    ))

    fig, ax = plt.subplots(nrows=3, ncols=5, subplot_kw={'adjustable': 'box'})

    image_comparison_filter_12_data, image_comparison_filter_12 = create_comparison(image_source_array[12],
                                                                                    image_mask_array[12])
    image_comparison_prediction_12_data, image_comparison_prediction_12 = create_comparison(image_prediction_12,
                                                                                            image_mask_array[12])

    image_comparison_filter_13_data, image_comparison_filter_13 = create_comparison(image_source_array[13],
                                                                                    image_mask_array[13])
    image_comparison_prediction_13_data, image_comparison_prediction_13 = create_comparison(image_prediction_13,
                                                                                            image_mask_array[13])

    image_comparison_filter_14_data, image_comparison_filter_14 = create_comparison(image_source_array[14],
                                                                                    image_mask_array[14])
    image_comparison_prediction_14_data, image_comparison_prediction_14 = create_comparison(image_prediction_14,
                                                                                            image_mask_array[14])

    print("\n\nfilter, 12: ")
    print_quality(image_comparison_filter_12_data)
    print("\nprediction, 12: ")
    print_quality(image_comparison_prediction_12_data)

    print("\n\nfilter, 13: ")
    print_quality(image_comparison_filter_13_data)
    print("\nprediction, 13: ")
    print_quality(image_comparison_prediction_13_data)

    print("\n\nfilter, 14: ")
    print_quality(image_comparison_filter_14_data)
    print("\nprediction, 14: ")
    print_quality(image_comparison_prediction_14_data)

    ax[0][0].imshow(image_mask_array[12])
    ax[0][1].imshow(image_source_array[12])
    ax[0][2].imshow(image_comparison_filter_12)
    ax[0][3].imshow(image_prediction_12)
    ax[0][4].imshow(image_comparison_prediction_12)

    ax[1][0].imshow(image_mask_array[13])
    ax[1][1].imshow(image_source_array[13])
    ax[1][2].imshow(image_comparison_filter_13)
    ax[1][3].imshow(image_prediction_13)
    ax[1][4].imshow(image_comparison_prediction_13)

    ax[2][0].imshow(image_mask_array[14])
    ax[2][1].imshow(image_source_array[14])
    ax[2][2].imshow(image_comparison_filter_14)
    ax[2][3].imshow(image_prediction_14)
    ax[2][4].imshow(image_comparison_prediction_14)

    save_img("mask_12.png", image_mask_array[12])
    save_img("filter_12.png", image_source_array[12])
    save_img("filter-mask_12.png", image_comparison_filter_12)
    save_img("prediction_12.png", image_prediction_12)
    save_img("prediction-mask_12.png", image_comparison_prediction_12)

    save_img("mask_13.png", image_mask_array[13])
    save_img("filter_13.png", image_source_array[13])
    save_img("filter-mask_13.png", image_comparison_filter_13)
    save_img("prediction_13.png", image_prediction_13)
    save_img("prediction-mask_13.png", image_comparison_prediction_13)

    save_img("mask_14.png", image_mask_array[14])
    save_img("filter_14.png", image_source_array[14])
    save_img("filter-mask_14.png", image_comparison_filter_14)
    save_img("prediction_14.png", image_prediction_14)
    save_img("prediction-mask_14.png", image_comparison_prediction_14)

    for a in ax:
        for b in a:
            b.axis('off')

    plt.tight_layout()
    fig.show()
    plt.show()


if __name__ == "__main__":
    main()
