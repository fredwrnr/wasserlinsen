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

folder = os.listdir()
if "output" not in folder:
    os.mkdir("output")

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


def read_all_images_in_folder(path=None):
    filenames = os.listdir(path)
    image_filenames = []
    if len(filenames) != 0:
        for file in filenames:
            if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png"):
                image_filenames.append(file)
    return image_filenames


def make_dataset(folder, with_bad = False):
    def on_mouse(event, x, y, flags, param):
        if param == "mark_plant":
            color = (0, 255, 0)
        elif param == "mark_background":
            color = (255, 0, 0)
        else:
            color = (0,0,255)

        if event == cv2.EVENT_LBUTTONDOWN or (event == cv2.EVENT_MOUSEMOVE and flags == cv2.EVENT_FLAG_LBUTTON):
            image[y-2:y+2,x-2:x+2] = color
            roi = cie_image[y-2:y+2,x-2:x+2].tolist()

            for row in roi:
                for pixel in row:
                    if param == "mark_plant":
                        plant_pixels.append(pixel)
                    elif param == "mark_background":
                        background_pixels.append(pixel)
                    else:
                        bad_plant_pixels.append(pixel)


    plant_pixels = []
    background_pixels = []
    bad_plant_pixels = []
    for i, filename in enumerate(read_all_images_in_folder(folder)):
        image = cv2.imread(os.path.join(folder, filename))
        image = image[pt1[1]:pt2[1], pt1[0]:pt2[0]]  # get only center
        image = cv2.resize(image, dsize=(image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

        cie_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        cv2.namedWindow("Bitte (gesunde) Wasserlinsen markieren", cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('Bitte (gesunde) Wasserlinsen markieren', on_mouse, "mark_plant")
        while True:
            cv2.imshow("Bitte (gesunde) Wasserlinsen markieren", image)
            if cv2.waitKey(1) & 0xFF == 32:
                break
        cv2.destroyAllWindows()

        if with_bad:
            cv2.namedWindow("Bitte tote Wasserlinsen markieren", cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback('Bitte tote Wasserlinsen markieren', on_mouse, "mark_bad")
            while True:
                cv2.imshow("Bitte tote Wasserlinsen markieren", image)
                if cv2.waitKey(1) & 0xFF == 32:
                    break
            cv2.destroyAllWindows()

        cv2.namedWindow("Bitte Hintergrund markieren", cv2.WINDOW_FULLSCREEN)
        cv2.setMouseCallback('Bitte Hintergrund markieren', on_mouse, "mark_background")
        while True:
            cv2.imshow("Bitte Hintergrund markieren", image)
            if cv2.waitKey(1) & 0xFF == 32:
                break
        cv2.destroyAllWindows()



    for i, pixel in enumerate(plant_pixels):
        plant_pixels[i] = pixel + [1]
    for i, pixel in enumerate(bad_plant_pixels):
        bad_plant_pixels[i] = pixel + [2]
    for i, pixel in enumerate(background_pixels):
        background_pixels[i] = pixel + [0]

    return plant_pixels + bad_plant_pixels + background_pixels


def train_svm(folder, with_bad):
    data = make_dataset(folder, with_bad)
    train, test = train_test_split(data, test_size=0.2, random_state=1)

    train_x = [row[:3] for row in train]  # Features: First three columns
    train_y = [row[3] for row in train]  # Labels: Fourth column
    test_x = [row[:3] for row in test]  # Features: First three columns
    test_y = [row[3] for row in test]  # Labels: Fourth column

    model = SVC(C=10, gamma=0.001, kernel="rbf")
    model.fit(train_x, train_y)

    pred = model.predict(test_x)
    rep = classification_report(test_y, pred, output_dict=True)
    print(rep)
    classes = 3 if with_bad else 2
    with open(f'svc_model_{classes}classes.sav', 'wb') as f:
        # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(model, f)
    return model, f'svc_model_{classes}classes.sav', rep["accuracy"]

def load_model(path):
    return pickle.load(open(path, 'rb'))


def run_inference(loaded_model, path):
    #loaded_model = pickle.load(open(model_path, 'rb'))
    print(path)
    image = cv2.imread(path)
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
            if result[0] == 1:
                image_mask[y, x] = [255, 0, 0]
            elif result[0] == 0:
                image_mask[y, x] = [0, 0, 0]
            elif result[0] == 2:
                image_mask[y, x] = [0, 0, 255]

    image_mask = delete_small_blobs(image_mask, 50)

    plant_area = np.count_nonzero(image_mask != 0)
    image_area = image.shape[0] * image.shape[1]
    plant_area_percentage = round(plant_area / image_area * 100, 2)
    print(plant_area_percentage)

    blended = cv2.addWeighted(image, 0.5, image_mask, 0.5, 0)

    stacked_image = np.concatenate([image, image_mask, blended], axis=1)
    cv2.imwrite(os.path.join("output", f"{path.split('.')[-2]}_output.jpg"), stacked_image)

    return cv2.cvtColor(stacked_image, cv2.COLOR_BGR2RGB), plant_area_percentage



if __name__ == "__main__":
    make_dataset()
    #train_svm()
    #main()