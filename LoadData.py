import os
import cv2

path = os.path.join(".", "dataset", "24 patient")
image_path = os.path.join("raw", "labeled")
mask_path = os.path.join("mask", "number")
image_save_path = os.path.join(".", "data", "images")
mask_save_path = os.path.join(".", "data", "masks")
counter = 0
for i in range(1, 25):
    patient = "patient" + str(i)
    images = os.path.join(path, patient, image_path)
    masks = os.path.join(path, patient, mask_path)
    for image, mask in zip(os.listdir(images), os.listdir(masks)):
        image_data = cv2.imread(os.path.join(images, image), 0)
        mask_data = cv2.imread(os.path.join(masks, mask), 0)
        cv2.imwrite(os.path.join(image_save_path, str(counter) + ".png"), image_data)
        cv2.imwrite(os.path.join(mask_save_path, str(counter) + ".png"), mask_data)
        counter += 1
    print(patient + " loaded.")

