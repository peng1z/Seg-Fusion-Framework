# segmentation/data_processing.py
# Helper functions for loading and preprocessing data for segmentation model training.

import os
import numpy as np
import pandas as pd
import shutil
from sklearn.model_selection import train_test_split
from skimage.io import imread
from skimage.transform import resize
from keras.preprocessing.image import img_to_array, load_img

def load_metadata_and_split_dataset(metadata_path, train_size=0.75, random_state=7):
    """Load metadata and split the dataset into train and test sets."""
    df_metadata = pd.read_csv(metadata_path)
    train_data, test_data = train_test_split(df_metadata, train_size=train_size, random_state=random_state, stratify=df_metadata.dx)
    return train_data, test_data

def copy_images_and_masks(source_folder, destination_folder, image_list):
    """Copy images from source folder to destination folder."""
    for image in image_list:
        shutil.copy(os.path.join(source_folder, image), destination_folder)

def preprocess_images(image_list, image_folder, mask_folder, image_size=(256, 384, 3)):
    """Preprocess images and masks."""
    X = np.zeros((len(image_list), *image_size), dtype=np.uint8)
    Y = np.zeros((len(image_list), *image_size[:2], 1), dtype=bool)
    
    for n, image_name in enumerate(image_list):
        # Process images
        img_path = os.path.join(image_folder, image_name)
        img = imread(img_path)[:,:,:image_size[2]]
        img_resized = resize(img, (image_size[0], image_size[1], image_size[2]), mode='constant', preserve_range=True)
        X[n] = img_resized
        
        # Process masks
        mask_path = os.path.join(mask_folder, image_name)
        mask = img_to_array(load_img(mask_path, color_mode='grayscale'))
        mask_resized = resize(mask, (image_size[0], image_size[1], 1), mode='constant', preserve_range=True)
        Y[n] = mask_resized
        
    return X, Y

def train_validation_split(X, Y, test_size=0.2, random_state=42):
    """Split the data into train and validation sets."""
    X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=test_size, random_state=random_state)
    return X_train, X_valid, y_train, y_valid
