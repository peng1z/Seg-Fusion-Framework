# segmentation/model_evaluation.py
# Evaluate the segmentation model.

from sklearn.metrics import mean_squared_error, f1_score, confusion_matrix, classification_report
import numpy as np

def evaluate_model(model, X_test, y_test, verbose=1):
    """Evaluate the segmentation model."""
    evaluation = model.evaluate(X_test, y_test, verbose=verbose)
    return evaluation

def dice_score(y_true, y_pred):
    """Calculate the Dice score."""
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection) / (np.sum(y_true) + np.sum(y_pred))
    return dice

def calculate_metrics(model, X_test, y_test):
    """Calculate additional evaluation metrics."""
    # Predict probabilities for test data
    y_pred_proba = model.predict(X_test)

    # Convert probabilities to binary predictions
    y_pred_binary = np.round(y_pred_proba).astype(int)

    # Calculate Mean Squared Error
    mse = mean_squared_error(y_test, y_pred_proba)
    print("Mean Squared Error:", mse)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred_binary)
    print("F1 Score:", f1)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred_binary)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate Dice score
    dice = dice_score(y_test, y_pred_binary)
    print("Dice Score:", dice)

    # Generate classification report
    class_report = classification_report(y_test, y_pred_binary)
    print("Classification Report:")
    print(class_report)

    return mse, f1, conf_matrix, dice, class_report
