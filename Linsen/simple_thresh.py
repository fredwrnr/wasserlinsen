import cv2
import numpy as np
import os
import csv

filenames = os.listdir()
if "output" not in filenames:
    os.mkdir("output")

image_filenames = []
if len(filenames) != 0:
    for file in filenames:
        if file.endswith(".jpg") or file.endswith(".jpeg"):
            image_filenames.append(file)
else:
    print("keine Bilder gefunden, Bilder im Format .bmp gespeichert?")
    exit()


if __name__ == "__main__":
    results = []
    for image_filename in image_filenames:
        # Load the image
        img = cv2.imread(image_filename, cv2.IMREAD_GRAYSCALE)

        img = cv2.medianBlur(img,5)


        mask = np.zeros_like(img)
        pt1 = (1150,580)
        pt2 = (3000,2410)
        mask = cv2.rectangle(mask, pt1, pt2, 255, -1)

        img[mask==0] = 255
        masked_img = img

        ret,th1 = cv2.threshold(masked_img,180,255,cv2.THRESH_BINARY)

        plant_area = np.count_nonzero(th1==0)
        rect_area = pt2[0]-pt1[0] * pt2[1] - pt1[1]
        plant_area_percentage = plant_area / rect_area * 100

        results.append((plant_area, round(plant_area_percentage,2)))


        org_image = cv2.imread(image_filename)
        org_image[th1 == 0] = (255,0,0)
        cv2.imwrite(os.path.join("output", f"{image_filename.split('.')[0]}_masked.jpg"), org_image)
        th1_small = cv2.resize(th1, (int(th1.shape[1]/4), int(th1.shape[0]/4)), interpolation= cv2.INTER_LINEAR)
        cv2.namedWindow("th1")
        cv2.imshow("th1", th1_small)
        cv2.waitKey(0)

    with open(os.path.join("output", 'output.csv'), mode='w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['Lemna Area [px]', 'Lemna Area [%]'])
        for image_entry in results:
            writer.writerow([*image_entry])