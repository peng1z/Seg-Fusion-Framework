# evaluation/model_evaluation.py
# Helper functions for evaluating the SegFusion framework.

import matplotlib.pyplot as plt

def plot_training_history(history):
    """
    Plot the training history (loss and accuracy).
    
    Args:
    - history: Training history returned by model.fit().
    """
    # Plot Training and Validation Loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

    # Plot Training and Validation Accuracy
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.show()
