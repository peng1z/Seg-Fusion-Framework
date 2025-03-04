# inference/data_preprocessing.py

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(images, labels):
    """
    Load image data and corresponding labels.
    
    Args:
    - images: Numpy array of images.
    - labels: Numpy array of labels.
    
    Returns:
    - Tuple (X_train, X_val, y_train, y_val): Train and validation data splits.
    """
    X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_val, y_train, y_val

def augment_data(images, labels, num_augmented_images=10):
    """
    Augment image data using ImageDataGenerator.
    
    Args:
    - images: Numpy array of images.
    - labels: Numpy array of labels.
    - num_augmented_images: Number of augmented images to generate per original image.
    
    Returns:
    - Tuple (augmented_images, augmented_labels): Augmented image data and labels.
    """
    # Define parameters for data augmentation
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True
    )

    # Initialize an empty list to store augmented images and labels
    augmented_images = []
    augmented_labels = []

    # Generate augmented images
    for image, label in zip(images, labels):
        # Reshape the image to 3D tensor (height, width, channels) for augmentation
        image = image.reshape((1,) + image.shape)
        # Generate augmented images
        augmented_batch = datagen.flow(image, batch_size=1)
        # Extract augmented images from the batch and corresponding labels
        augmented_images.extend([next(augmented_batch)[0] for _ in range(num_augmented_images)])
        augmented_labels.extend([label] * num_augmented_images)  # Maintain consistency in labels

    # Convert lists of augmented images and labels to numpy arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)

    return augmented_images, augmented_labels
