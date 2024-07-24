import os
import cv2
import numpy as np
from keras.utils import to_categorical
# Define the path to directories containing images

# Function to get label from image name


# Function to read images and labels from directory
def read_images_and_labels(llll,classssss):
    images = []
    labels = []
    for filename, image_label in zip(llll, classssss):
        print(filename, image_label)
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = filename
            label = image_label
            img = cv2.imread(img_path)
            print(img.size)
            resized_img = cv2.resize(img, (75, 250))   # resize the image to (250, 75)
            images.append(resized_img)
            labels.append(label)
    labels = to_categorical(labels, 54)
    return (np.array(images), np.array(labels))

