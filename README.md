# Seg-Fusion-Framework

## Introduction

We proposes a SegFusion framework and aims to develop and deploy machine learning models for the segmentation and classification of skin cancer images. This study leverages deep learning techniques to assist medical professionals in accurately diagnosing skin cancer from dermatoscopic images. Paper is [here](https://arxiv.org/pdf/2408.00772) and has been accepted by BIOKDD'24.

## Structure

The repo is organized into two main modules. Several sub-modules responsible for data preprocessing, model building, model training, and model evaluation.

1. **Segmentation Module**: This module focuses on segmenting skin lesions from dermatoscopic images using `U-Net`, a convolutional neural network architecture widely used for medical image segmentation tasks.
2. **Classification Training Module**: This module focuses on classifying skin cancer images into benign and malignant categories. It uses transfer learning with pre-trained `EfficientNet-B0` model to achieve high accuracy in classification.

This study employs two key data processing/inference techniques.

- Applying the masks to the original images before passing them into the classification model. Below is the visualization.

- Augmenting data includes rotation, width shift, height shift, zoom, horizontal flip, and vertical flip. Below is an example.

## Performance Metrics
The SegFusion method achieved 99.01% accuracy on the ISIC 2020 dataset, 98.93% accuracy on the ISIC 2019 dataset, and 98.21% accuracy on the HAM10000 dataset. More metrics is shown in below table.

```
Datasets    Accuracy   Precision   Recall   F1 Score   MCC

ISIC 2020    99.01%      0.99       0.99      0.99     0.97

ISIC 2019    98.93%      0.99       0.98      0.99     0.98

HAM10000     98.21%      0.98       0.98      0.98     0.96
```

- ROC-AUC:
<img src="images/Receiver%20Operating%20Characteristic%20ISIC2020.png" width="400" alt="ROC_AUC_curve">

- PR-AUC:
<img src="images/Precision-Recall%20Curve%20ISIC2020.png" width="400" alt="PR_AUC_curve">

The high F1 score, accuracy, and precision-recall values indicate that the SegFusion framework perform well in detecting and classifying skin lesions with an accuracy of **99.01%**, making them valuable tools for medical diagnosis and research.

## Quick Start

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Download the pre-trained model weights for UNet and MobileNetV3 and place them in the appropriate folders.

## Usage

### Segmentation/Classification Module

1. **Data Preprocessing**: Use `data_preprocessing.py` to load, preprocess, and augment the segmentation data.
2. **Model Building**: Use `model_building.py` to define the U-Net architecture for segmentation.
3. **Model Training**: Use `model_training.py` to train the segmentation model on the provided dataset.
4. **Model Evaluation**: Use `model_evaluation.py` to evaluate the segmentation model's performance and visualize the segmentation results.

## Dependencies

- Python 3.x
- TensorFlow
- scikit-learn
- matplotlib
- pandas
- numpy

## Acknowledgements

- This study utilizes the HAM10000 and ISIC 2020 datasets for training and evaluation.
- Special thanks to the developers and contributors of TensorFlow, scikit-learn, and other open-source libraries used in this study.