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


class LemnaMaster:
    def __init__(self, image_outline_points: dict):

        # self.pt1 = (1150,580)    # x,y
        # self.pt2 = (3000,2410)
        self.pt1 = image_outline_points["p1"]
        self.pt2 = image_outline_points["p2"]
        print(self.pt1)
        print(self.pt2)
        self.loaded_model = None

    def load_model(self, model_path):
        self.loaded_model = pickle.load(open(model_path, 'rb'))

    @staticmethod
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


    def read_all_images_in_folder(self, path=None):
        filenames = os.listdir(path)
        image_filenames = []
        if len(filenames) != 0:
            for file in filenames:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".JPG") or file.endswith(".PNG"):
                    image_filenames.append(file)
        return image_filenames


    def make_dataset(self, folder, with_bad = False):
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
        for i, filename in enumerate(self.read_all_images_in_folder(folder)):
            image = cv2.imread(os.path.join(folder, filename))
            image = image[self.pt1[1]:self.pt2[1], self.pt1[0]:self.pt2[0]]  # get only center
            image = cv2.resize(image, dsize=(image.shape[1] // 2, image.shape[0] // 2), interpolation=cv2.INTER_CUBIC)

            cie_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

            cv2.namedWindow("Bitte (gesunde) Wasserlinsen markieren", cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('Bitte (gesunde) Wasserlinsen markieren', on_mouse, "mark_plant")
            while True:
                cv2.imshow("Bitte (gesunde) Wasserlinsen markieren", image)
                if cv2.waitKey(1) & 0xFF == 32:
                    break
            cv2.destroyAllWindows()

            if with_bad:
                cv2.namedWindow("Bitte tote Wasserlinsen markieren", cv2.WINDOW_NORMAL)
                cv2.setMouseCallback('Bitte tote Wasserlinsen markieren', on_mouse, "mark_bad")
                while True:
                    cv2.imshow("Bitte tote Wasserlinsen markieren", image)
                    if cv2.waitKey(1) & 0xFF == 32:
                        break
                cv2.destroyAllWindows()

            cv2.namedWindow("Bitte Hintergrund markieren", cv2.WINDOW_NORMAL)
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


    def train_svm(self, folder, with_bad):
        data = self.make_dataset(folder, with_bad)
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

    def run_inference_on_image(self, image_path):
        image = cv2.imread(image_path)
        image = image[self.pt1[1]:self.pt2[1], self.pt1[0]:self.pt2[0]]  # get only center
        image = cv2.resize(image, dsize=(image.shape[1] // 4, image.shape[0] // 4), interpolation=cv2.INTER_CUBIC)
        cie_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

        # Convert image to a list of pixels for batch prediction
        pixels = cie_image.reshape(-1, 3)

        # Perform batch prediction instead of pixel-by-pixel
        results = self.loaded_model.predict(pixels)

        # Create the mask using numpy operations instead of pixel-by-pixel assignments
        image_mask = np.zeros_like(cie_image)

        # Reshape results back to image dimensions
        results = results.reshape(cie_image.shape[0], cie_image.shape[1])

        # Use boolean indexing for faster assignments
        image_mask[results == 1] = [255, 0, 0]
        image_mask[results == 0] = [0, 0, 0]
        image_mask[results == 2] = [0, 0, 255]

        image_mask = self.delete_small_blobs(image_mask, 20)

        # Use numpy's sum and boolean operations for faster area calculations
        image_area = image_mask.shape[0] * image_mask.shape[1]
        plant_mask = np.all(image_mask == [255, 0, 0], axis=-1)
        dead_mask = np.all(image_mask == [0, 0, 255], axis=-1)
        background_mask = np.all(image_mask == [0, 0, 0], axis=-1)

        plant_area = np.sum(plant_mask)
        dead_plant_area = np.sum(dead_mask)
        background_area = np.sum(background_mask)

        plant_area_percentage = round(plant_area / image_area * 100, 2)
        dead_plant_area_percentage = round(dead_plant_area / image_area * 100, 2)
        background_area_percentage = round(background_area / image_area * 100, 2)

        print(f"plant_area_percentage: {plant_area_percentage}")
        print(f"dead_plant_area_percentage: {dead_plant_area_percentage}")
        print(f"background_area_percentage: {background_area_percentage}")

        blended = cv2.addWeighted(image, 0.5, image_mask, 0.5, 0)
        stacked_image = np.concatenate([image, image_mask, blended], axis=1)

        self.save_inference_result(image_path, stacked_image, plant_area_percentage, dead_plant_area_percentage)

        small_stacked_image = cv2.resize(stacked_image, dsize=(stacked_image.shape[1] // 2, stacked_image.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
        return cv2.cvtColor(small_stacked_image, cv2.COLOR_BGR2RGB), plant_area_percentage, dead_plant_area_percentage

    def save_inference_result(self, image_path, stacked_image, plant_area_percentage, dead_area_percentage):
        os.makedirs("output", exist_ok=True)
        image_name = os.path.split(image_path.split('.')[-2])[-1]
        cv2.imwrite(os.path.join("output", f"{image_name}_output.jpg"), stacked_image)
        if not os.path.isfile(os.path.join("output", "output.csv")):
            with open(os.path.join("output", 'output.csv'), mode='w', newline='') as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(['Filename', 'Lemna Area [%]', 'Dead Lemna Area [%]'])

        with open(os.path.join("output", 'output.csv'), mode='a', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            print(f"saving results: {plant_area_percentage}")
            writer.writerow([image_name, plant_area_percentage, dead_area_percentage])


if __name__ == "__main__":
    pass
    #make_dataset()
    #train_svm()
    #main()