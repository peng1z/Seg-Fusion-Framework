# segmentation/model building.py
# Define the UNET model for image segmentation.

from keras.models import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, Dropout, concatenate, BatchNormalization, Activation
from keras.optimizers import Adam

def get_conv2d_layers(input_tensor, n_filters, kernel_size=3, batchnorm=True):
    """Function to create convolution layers with the given input parameters."""
    # Layer 1
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),\
              kernel_initializer='he_normal', padding='same')(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Layer 2
    x = Conv2D(filters=n_filters, kernel_size=(kernel_size, kernel_size),\
              kernel_initializer='he_normal', padding='same')(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def unet(input_img, n_filters=16, dropout=0.1, batchnorm=True):
    """Function to define the UNET Model."""
    # UNET Contracting path
    conv1 = get_conv2d_layers(input_img, n_filters * 1, kernel_size=3, batchnorm=batchnorm)
    pool1 = MaxPooling2D((2, 2))(conv1)
    pool1 = Dropout(dropout)(pool1)

    conv2 = get_conv2d_layers(pool1, n_filters * 2, kernel_size=3, batchnorm=batchnorm)
    pool2 = MaxPooling2D((2, 2))(conv2)
    pool2 = Dropout(dropout)(pool2)

    conv3 = get_conv2d_layers(pool2, n_filters * 4, kernel_size=3, batchnorm=batchnorm)
    pool3 = MaxPooling2D((2, 2))(conv3)
    pool3 = Dropout(dropout)(pool3)

    conv4 = get_conv2d_layers(pool3, n_filters * 8, kernel_size=3, batchnorm=batchnorm)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout)(pool4)

    conv5 = get_conv2d_layers(pool4, n_filters=n_filters * 16, kernel_size=3, batchnorm=batchnorm)

    # UNET Expansive path
    up6 = Conv2DTranspose(n_filters * 8, (3, 3), strides=(2, 2), padding='same')(conv5)
    up6 = concatenate([up6, conv4])
    up6 = Dropout(dropout)(up6)
    conv6 = get_conv2d_layers(up6, n_filters * 8, kernel_size=3, batchnorm=batchnorm)

    up7 = Conv2DTranspose(n_filters * 4, (3, 3), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, conv3])
    up7 = Dropout(dropout)(up7)
    conv7 = get_conv2d_layers(up7, n_filters * 4, kernel_size=3, batchnorm=batchnorm)

    up8 = Conv2DTranspose(n_filters * 2, (3, 3), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, conv2])
    up8 = Dropout(dropout)(up8)
    conv8 = get_conv2d_layers(up8, n_filters * 2, kernel_size=3, batchnorm=batchnorm)

    up9 = Conv2DTranspose(n_filters * 1, (3, 3), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, conv1])
    up9 = Dropout(dropout)(up9)
    conv9 = get_conv2d_layers(up9, n_filters * 1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)
    model = Model(inputs=[input_img], outputs=[outputs])
    return model
