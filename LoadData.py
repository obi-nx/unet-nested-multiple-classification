import os
import cv2 as cv
from config import DATASET, RAW_DATA, CLAHE_DATA, NLMEANS_DATA, ESRGAN_DATA
from os.path import join


def ready_up_folder(folder):
    image_save_path = join(folder, "images")
    mask_save_path = join(folder, "masks")
    os.makedirs(folder, exist_ok=True)
    os.makedirs(image_save_path, exist_ok=True)
    os.makedirs(mask_save_path, exist_ok=True)
    return image_save_path, mask_save_path


path = join(DATASET, "24 patient")
image_path = join("raw", "labeled")
mask_path = join("mask", "number")
raw_images, raw_masks = ready_up_folder(RAW_DATA)
clahe_images, clahe_masks = ready_up_folder(CLAHE_DATA)
nlmeans_images, nlmeans_masks = ready_up_folder(NLMEANS_DATA)
esrgan_images, esrgan_masks = ready_up_folder(ESRGAN_DATA)
counter = 0
for patient in os.listdir(path):
    images = join(path, patient, image_path)
    masks = join(path, patient, mask_path)
    for image, mask in zip(os.listdir(images), os.listdir(masks)):
        # Load original data
        image_data = cv.imread(join(images, image), 0)
        mask_data = cv.imread(join(masks, mask), 0)

        # Raw data
        cv.imwrite(join(raw_images, str(counter) + ".png"), image_data)
        cv.imwrite(join(raw_masks, str(counter) + ".png"), mask_data)

        # CLAHE applied data
        # TODO: Parameter optimization
        clahe_obj = cv.createCLAHE()
        clahe_image = clahe_obj.apply(image_data)
        cv.imwrite(join(clahe_images, str(counter) + ".png"), clahe_image)
        cv.imwrite(join(clahe_masks, str(counter) + ".png"), mask_data)

        # NLMEANS applied data
        # TODO: Parameter optimization
        nlmeans_image = cv.fastNlMeansDenoising(image_data)
        cv.imwrite(join(nlmeans_images, str(counter) + ".png"), nlmeans_image)
        cv.imwrite(join(nlmeans_masks, str(counter) + ".png"), mask_data)

        # ESRGAN applied data
        # TODO: Implement this method!

        counter += 1
    print(patient + " loaded.")
