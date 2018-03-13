import os
import numpy as np
import cv2
import matplotlib.pyplot as plt 

from config import *

def create_mask_for_plant(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sensitivity = 35
    lower_hsv = np.array([60 - sensitivity, 100, 50])
    upper_hsv = np.array([60 + sensitivity, 255, 255])

    mask = cv2.inRange(image_hsv, lower_hsv, upper_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    return mask

def segment_plant(image):
    mask = create_mask_for_plant(image)
    output = cv2.bitwise_and(image, image, mask = mask)
    return output

def sharpen_image(image):
    image_blurred = cv2.GaussianBlur(image, (0, 0), 3)
    image_sharp = cv2.addWeighted(image, 1.5, image_blurred, -0.5, 0)
    return image_sharp


if __name__ == '__main__':
    for index, label in enumerate(labels):
        folder_path = os.path.join('train', label)
        for file in os.listdir(folder_path):
            image_path = os.path.join(folder_path, file)
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image_segmented = segment_plant(image_bgr)
            image_sharpen = sharpen_image(image_segmented)
            save_path = os.path.join('seg_train', label)
            if not os.path.isdir(save_path):
                os.mkdir(save_path)
            cv2.imwrite(os.path.join(save_path, file), image_sharpen)

    for file in os.listdir('test'):
        image_path = os.path.join('test', file)
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image_segmented = segment_plant(image_bgr)
        image_sharpen = sharpen_image(image_segmented)
        cv2.imwrite(os.path.join('seg_test', file), image_sharpen)