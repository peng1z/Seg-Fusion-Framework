# training/model_building.py
# Build a classification model using EfficientNet-B0 architecture.

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

def build_classification_model(input_shape):
    """
    Build a classification model using EfficientNet-B0 architecture.
    
    Args:
    - input_shape: Tuple specifying the input shape (height, width, channels).
    
    Returns:
    - classify_model: Compiled classification model.
    """
    # Load EfficientNet-B0 model
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')

    # Add classification layers
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.1)(x)
    output = Dense(1, activation='sigmoid')(x)

    # Create classification model
    classify_model = Model(inputs=base_model.input, outputs=output)

    # Compile classification model
    classify_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return classify_model
