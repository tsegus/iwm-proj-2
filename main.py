# wstępne przetworzenie
# wejściowy obraz może być tutaj za jasny itd.
# właściwe przetworzenie
# wykrywanie krawędzi np. filtr frangiego
# końcowe przetworzenie
# przetwarzenie uzyskanego obrazu i jego naprawa

from skimage.io import ImageCollection, imread
from skimage.filters import frangi
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier


def main():

    # zbiór zdjęć
    eye_photo_list = []
    eye_fov_list = []
    eye_mask_list = []

    for i in range(1, 16):
        if i < 10:
            number = "0" + str(i)
        else:
            number = str(i)

        eye_photo_list.append("data/healthy/" + number + "_h.jpg")
        eye_fov_list.append("data/healthy_fovmask/" + number + "_h_mask.tif")
        eye_mask_list.append("data/healthy_manualsegm/" + number + "_h.tif")

    print("opening photos...")

    img_eye_photo = []
    img_eye_fov = []
    img_eye_mask = []

    for i in range(15):
        img_eye_photo.append(imread(eye_photo_list[i]))
        img_eye_fov.append(imread(eye_fov_list[i], as_gray=True))
        img_eye_mask.append(imread(eye_mask_list[i], as_gray=True))

    print("sharpening...")

    """ Sharpening """
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    for i in range(15):
        img_eye_photo[i] = cv2.filter2D(src=img_eye_photo[i], ddepth=-1, kernel=kernel)

    img_eye_photo_filtered = []

    print("filtering...")

    """ Filtering """
    for i in range(15):
        img_eye_photo_filtered.append(frangi(rgb2gray(img_eye_photo[i])))

    print("bitmap...")

    """ Bitmap """
    for i in range(15):
        bitmap = img_eye_photo_filtered[i]
        fov = img_eye_fov[i]

        thresholdValue = np.mean(bitmap)
        xDim, yDim = bitmap.shape
        for j in range(xDim):
            for k in range(yDim):
                if fov[j][k] > 0:
                    if bitmap[j][k] > thresholdValue / 2:
                        bitmap[j][k] = 255
                    else:
                        bitmap[j][k] = 0
        img_eye_photo_filtered[i] = bitmap

    print("comparing...")

    """ obraz porównawczy """
    compare_img_eye_photo_filtered = []

    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    for i in range(15):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        photo = img_eye_photo_filtered[i]
        mask = img_eye_mask[i]
        xDim, yDim = photo.shape

        compare = []

        for j in range(xDim):
            compare_row = []
            for k in range(yDim):
                if photo[j][k] == 0 and mask[j][k] == 0:
                    TN += 1
                    compare_row.append([0, 0, 0])
                elif photo[j][k] == 0 and mask[j][k] == 255:
                    FN += 1
                    compare_row.append([255, 255, 0])
                elif photo[j][k] == 255 and mask[j][k] == 0:
                    FP += 1
                    compare_row.append([255, 0, 0])
                else:
                    TP += 1
                    compare_row.append([255, 255, 255])
            compare.append(compare_row)

        compare_img_eye_photo_filtered.append(compare)

        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)

    print("calculating error measures...")

    accuracy_list = []
    sensitivity_list = []
    specificity_list = []

    for i in range(15):

        TP = TP_list[i]
        FP = FP_list[i]
        TN = TN_list[i]
        FN = FN_list[i]

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    print("filter avg accuracy: " + str(100 * np.mean(accuracy_list)) + "%")
    print("filter avg sensitivity: " + str(100 * np.mean(sensitivity_list)) + "%")
    print("filter avg specificity: " + str(100 * np.mean(specificity_list)) + "%")

    print("knn...")

    classifier = KNeighborsClassifier(
        n_neighbors=5,
        weights='uniform',
        algorithm='auto',
        leaf_size=20,
        p=2,
    )

    for i in range(12):
        classifier.fit(
            img_eye_photo_filtered[i],
            img_eye_mask[i]
        )

    #classifier.fit

    print("predictions...")

    knn_predictions = []

    for i in range(12, 15):
        knn_predictions.append(classifier.predict(
            img_eye_photo_filtered[i]
        ))

    """ obraz porównawczy """
    compare_img_eye_photo_knn = []

    TP_list = []
    FP_list = []
    TN_list = []
    FN_list = []

    for i in range(3):
        TP = 0
        FP = 0
        TN = 0
        FN = 0

        photo = knn_predictions[i]
        mask = img_eye_mask[i+12]
        xDim, yDim = photo.shape

        compare = []

        for j in range(xDim):
            compare_row = []
            for k in range(yDim):
                if photo[j][k] == 0 and mask[j][k] == 0:
                    TN += 1
                    compare_row.append([0, 0, 0])
                elif photo[j][k] == 0 and mask[j][k] == 255:
                    FN += 1
                    compare_row.append([255, 255, 0])
                elif photo[j][k] == 255 and mask[j][k] == 0:
                    FP += 1
                    compare_row.append([255, 0, 0])
                else:
                    TP += 1
                    compare_row.append([255, 255, 255])
            compare.append(compare_row)

        compare_img_eye_photo_knn.append(compare)

        TP_list.append(TP)
        FP_list.append(FP)
        TN_list.append(TN)
        FN_list.append(FN)

    print("calculating error measures...")

    accuracy_list = []
    sensitivity_list = []
    specificity_list = []

    for i in range(3):

        TP = TP_list[i]
        FP = FP_list[i]
        TN = TN_list[i]
        FN = FN_list[i]

        accuracy = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)

        accuracy_list.append(accuracy)
        sensitivity_list.append(sensitivity)
        specificity_list.append(specificity)

    print("knn avg accuracy: " + str(100 * np.mean(accuracy_list)) + "%")
    print("knn avg sensitivity: " + str(100 * np.mean(sensitivity_list)) + "%")
    print("knn avg specificity: " + str(100 * np.mean(specificity_list)) + "%")


    print("display...")

    """ Display """
    fig = plt.figure()

    ax1 = fig.add_subplot(2, 3, 1)
    ax1.imshow(compare_img_eye_photo_filtered[0])

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.imshow(compare_img_eye_photo_filtered[1])

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.imshow(compare_img_eye_photo_filtered[2])

    ax4 = fig.add_subplot(2, 3, 4)
    ax4.imshow(compare_img_eye_photo_knn[0])

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.imshow(compare_img_eye_photo_knn[1])

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.imshow(compare_img_eye_photo_knn[2])

    fig.show()
    plt.show()


if __name__ == "__main__":
    main()
