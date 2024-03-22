import numpy as np
import csv
import pickle
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
import os
import cv2
from tqdm import tqdm
import time


pt1 = (1150,580)    # x,y
pt2 = (3000,2410)


def delete_small_blobs(image, size_threshold):
    """
    Deletes blobs in an image that are smaller than a certain threshold.
    Blobs are identified as connected regions with pixel values [255, 0, 0].
    The background is assumed to be [0, 0, 0].

    Args:
    - image (np.ndarray): The input image as a NumPy array of shape (height, width, 3).
    - size_threshold (int): The size threshold for blob deletion.

    Returns:
    - np.ndarray: The processed image with small blobs removed.
    """
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Create a binary mask where red blobs are white and the rest is black
    red_blobs_mask = np.all(image == [255, 0, 0], axis=-1).astype(np.uint8) * 255

    # Find connected components in the binary mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(red_blobs_mask, 8, cv2.CV_32S)

    # Create an output image that starts as a copy of the original
    output_image = image.copy()

    # Iterate through found components, excluding the background (label 0)
    for label in range(1, num_labels):
        # If the component size is smaller than the threshold, delete it
        if stats[label, cv2.CC_STAT_AREA] < size_threshold:
            # Set the pixels corresponding to this component to the background color
            output_image[labels == label] = [0, 0, 0]

    return output_image


def read_all_images_in_folder():
    filenames = os.listdir()
    image_filenames = []
    if len(filenames) != 0:
        for file in filenames:
            if file.endswith(".jpg") or file.endswith(".jpeg"):
                image_filenames.append(file)
    return image_filenames


def make_dataset():
    drawing = False
    pt1_x, pt1_y = None, None
    def on_mouse(event, x, y, flags, param):
        global pt1_x,pt1_y,drawing
        if param == "mark_plant":
            color = (0, 255, 0)
        else:
            color = (255, 0, 0)

        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
            image[y-2:y+2,x-2:x+2] = color
            roi = cie_image[y-2:y+2,x-2:x+2].tolist()

            for row in roi:
                for pixel in row:
                    if param == "mark_plant":
                        plant_pixels.append(pixel)
                    else:
                        background_pixels.append(pixel)


    plant_pixels = []
    background_pixels = []
    for i, filename in enumerate(read_all_images_in_folder()):
        if filename != "image0.jpeg" and filename != "image3.jpeg":
            continue

        image = cv2.imread(filename)
        image = cv2.resize(image, dsize=(image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_CUBIC)
        image = image[pt1[1] // 4:pt2[1] // 4, pt1[0] // 4:pt2[0] // 4]
        cie_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        cv2.namedWindow("mark_plant", cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('mark_plant', on_mouse, "mark_plant")
        while True:
            cv2.imshow("mark_plant", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

        cv2.namedWindow("mark_background", cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('mark_background', on_mouse, "mark_background")
        while True:
            cv2.imshow("mark_background", image)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    for i, pixel in enumerate(plant_pixels):
        plant_pixels[i] = pixel + [1]
    for i, pixel in enumerate(background_pixels):
        background_pixels[i] = pixel + [0]
    with open(f'dataset_test.csv', "w", newline='') as file:
        write = csv.writer(file)
        write.writerows(plant_pixels)
        write.writerows(background_pixels)


def train_svm():
    data = []
    with open("dataset.csv", mode='r', encoding='utf-8') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            data.append(row)

    train, test = train_test_split(data, test_size=0.2, random_state=1)

    train_x = [row[:3] for row in train]  # Features: First three columns
    train_y = [row[3] for row in train]  # Labels: Fourth column
    test_x = [row[:3] for row in test]  # Features: First three columns
    test_y = [row[3] for row in test]  # Labels: Fourth column

    model = SVC(C=10, gamma=0.001, kernel="rbf")
    model.fit(train_x, train_y)

    pred = model.predict(test_x)
    print(classification_report(test_y, pred))

    with open(f'svc_model.sav', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(model, f)


def main():
    loaded_model = pickle.load(open("svc_model.sav", 'rb'))

    for i, filename in enumerate(read_all_images_in_folder()):
        print(filename)
        image = cv2.imread(filename)
        image = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]  # get only center
        image = cv2.resize(image, dsize=(image.shape[1] // 8, image.shape[0] // 8), interpolation=cv2.INTER_CUBIC)
        cie_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        width = cie_image.shape[1]
        height = cie_image.shape[0]
        image_mask = cie_image.copy()
        for x in tqdm(range(width)):
            for y in range(height):
                pixel = [cie_image[y, x]]
                result = loaded_model.predict(pixel)
                if result == "1":

                    image_mask[y, x] = [255, 0, 0]
                else:
                    image_mask[y, x] = [0, 0, 0]

        image_mask = delete_small_blobs(image_mask, 50)
        blended = cv2.addWeighted(image, 0.5, image_mask, 0.5, 0)
        cv2.namedWindow("org_image", cv2.WINDOW_NORMAL)
        cv2.imshow("org_image", image)
        cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
        cv2.imshow("mask", image_mask)
        cv2.namedWindow("blended", cv2.WINDOW_NORMAL)
        cv2.imshow("blended", blended)
        cv2.waitKey(0)


if __name__ == "__main__":
    make_dataset()
    #train_svm()
    #main()