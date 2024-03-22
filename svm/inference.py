import numpy as np
import pickle
import cv2
import time
from tqdm import tqdm
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import math
import os


class InferenceModel:
    def __init__(self, camera_height):
        self.model = pickle.load(open("basil_svc_model.sav", 'rb'))
        self.camera_height = camera_height

    @staticmethod
    def find_plant(binary_mask):
        binary_mask2D = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        image_center_coords = (int(binary_mask.shape[1]/2), int(binary_mask.shape[0]/2))

        output = cv2.connectedComponentsWithStats(binary_mask2D, 8, cv2.CV_32S)
        (numLabels, pix_labels, stats, centroids) = output  # labels from 0 to i

        # loop over the number of unique connected component labels
        for i in range(1, numLabels):
            # i=0 is background
            area = stats[i, cv2.CC_STAT_AREA]
            (cX, cY) = centroids[i]
            distance_from_center = math.dist(image_center_coords, (cX, cY))
            if distance_from_center > 200:  # and image center not in label area
                binary_mask[pix_labels == i] = [0,0,0]

        return binary_mask

    def get_mask(self, img):
        # convert image to cie-colorspace (has to be same colorspace as trained model!)
        lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)

        t0 = time.time()
        org_width = img.shape[1]
        org_height = img.shape[0]
        resized_image = cv2.resize(lab_img, dsize=(int(org_width / 4), int(org_height / 4)),
                                   interpolation=cv2.INTER_CUBIC)
        reduced_width = resized_image.shape[1]
        reduced_height = resized_image.shape[0]
        resized_binary_mask = resized_image.copy()
        for x in tqdm(range(reduced_width)):
            for y in range(reduced_height):
                px = [resized_image[y, x]]
                resized_binary_mask[y, x] = self.model.predict(px)
        binary_mask = cv2.resize(resized_binary_mask, dsize=(org_width, org_height), interpolation=cv2.INTER_CUBIC)
        print(f"inference_time: {time.time() - t0}")
        if np.count_nonzero(binary_mask == 1) < 100:
            binary_mask = None

        plant_binary_mask = self.find_plant(binary_mask)

        return plant_binary_mask  # mask is 3d-format [0,0,0] or [1,1,1]

    @staticmethod
    def calc_area(binary_mask_3d, img):
        # greenness with white balance
        binary_mask_2d = cv2.cvtColor(binary_mask_3d, cv2.COLOR_BGR2GRAY)
        total_leaf_area = np.count_nonzero(binary_mask_2d == 1)
        points = []
        for i in range(len(binary_mask_2d)):
            for j in range(len(binary_mask_2d[i])):
                if binary_mask_2d[i, j] == 1:
                    points.append((j, i))

        try:
            hull = ConvexHull(points)
            x_coords = []
            y_coords = []
            hull_img = img.copy()

            for simplex in hull.simplices:
                point_1 = points[simplex[0]]
                point_2 = points[simplex[1]]
                hull_img = cv2.line(hull_img, point_1, point_2, color=[255, 100, 100], thickness=4)
            for vertix in hull.vertices:
                x_coords.append(points[vertix][0])
                y_coords.append(points[vertix][1])
            convex_hull_area = int(
                0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1))))
            solidity = total_leaf_area / convex_hull_area
        except IndexError:
            hull_img = img
            convex_hull_area = 0
            solidity = 0

        return {"total_leaf_area": total_leaf_area, "convex_hull_area": convex_hull_area, "solidity": solidity}, hull_img

    def calc_height_values(self, binary_mask, img, depth_map):
        # convert depth_map to height_map
        height_map = self.camera_height - depth_map
        # convert from 3D to 2D
        binary_mask_2d = cv2.cvtColor(binary_mask, cv2.COLOR_BGR2GRAY)
        # multiply each value with the binary mask, so values outside the mask will be set to 0
        masked_height_map = height_map * binary_mask_2d
        # filter out invalid values (values that used to be 0 and are now 660)
        masked_height_map[masked_height_map == self.camera_height] = 0
        # set all 0-values to NaN, so that they don't affect the average and median
        masked_height_map = masked_height_map.astype('float')
        masked_height_map[masked_height_map == 0] = np.NaN

        average_height = int(np.nanmean(masked_height_map))
        max_height = int(np.nanmax(masked_height_map))
        median_height = int(np.nanmedian(masked_height_map))
        return {"average_height": average_height, "max_height": max_height, "median_height": median_height}

    def run_inference(self, img, depth_map=None, show_imgs=False):
        binary_mask = self.get_mask(img)
        if binary_mask is None:
            print("no plant found in image")
            return False
        else:
            blended_image = img.copy()
            blended_image[(binary_mask == 1).all(-1)] = [255, 0, 0]

            if show_imgs:
                depth_image = cv2.applyColorMap(cv2.convertScaleAbs(depth_map, alpha=0.3), cv2.COLORMAP_JET)
                images = np.hstack((img, blended_image, depth_image))
                # Show images
                cv2.namedWindow('Stacked', cv2.WINDOW_AUTOSIZE)
                cv2.imshow('Stacked', images)
                # cv2.imwrite(f"image{i}.png", color_image)
                # cv2.imwrite(f"color{i}.bmp", depth_image)
                key = cv2.waitKey(0)
                if key & 0xFF == ord('q') or key == 27:
                    cv2.destroyAllWindows()

            area_dict, hull_img = self.calc_area(binary_mask, img)
            if depth_map is not None:
                height_dict = self.calc_height_values(binary_mask, img, depth_map)
            else:
                height_dict = None
            return binary_mask, blended_image, hull_img, area_dict, height_dict


if __name__ == "__main__":
    InferenceModel = InferenceModel(camera_height=int(542))
    test_images_path = "/Users/fred/Dropbox/Acheron/Testbilder_pakchoi"
    for filename in os.listdir(test_images_path):
        test_image = cv2.imread(f"{test_images_path}/{filename}")
        binary_mask, blended_image, hull_img, area_dict, height_dict = InferenceModel.run_inference(test_image)
        cv2.namedWindow("blended", cv2.WINDOW_NORMAL)
        cv2.imshow("blended", blended_image)
        cv2.namedWindow("hull_img", cv2.WINDOW_NORMAL)
        cv2.imshow("hull_img", hull_img)
        cv2.waitKey(0)
        print(area_dict)
        print(height_dict)
