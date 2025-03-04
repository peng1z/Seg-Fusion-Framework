# training/model_training.py
# Helper functions for training classification models.

def train_classification_model(classify_model, X_train, y_train, X_val, y_val, batch_size=32, epochs=50):
    """
    Train the classification model.
    
    Args:
    - classify_model: Compiled classification model.
    - X_train: Training images.
    - y_train: Training labels.
    - X_val: Validation images.
    - y_val: Validation labels.
    - batch_size: Batch size for training.
    - epochs: Number of epochs for training.
    
    Returns:
    - history: Training history.
    """
    history = classify_model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        verbose=1
    )
    return history
