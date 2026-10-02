# Handwritten Character Recognition using CNN

## Project Overview

This project was completed as part of the CodeAlpha Machine Learning Internship.

The objective of this project is to recognize handwritten digits using a Convolutional Neural Network (CNN). The model is trained and evaluated on the MNIST handwritten digit dataset.

The CNN learns visual patterns from 28×28 grayscale images and classifies each image into one of the ten digit classes (0–9).

---

## Dataset

Dataset: MNIST Handwritten Digits

Source: Kaggle - MNIST in CSV

Dataset files used:

- `mnist_train.csv`
- `mnist_test.csv`

Dataset size:

- Training samples: 60,000
- Testing samples: 10,000
- Image size: 28 × 28 pixels
- Number of classes: 10 (digits 0–9)
- Image type: Grayscale

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the training and testing CSV files using Pandas.
2. Separated the `label` column from the pixel features.
3. Normalized pixel values from the range 0–255 to 0–1.
4. Reshaped the images from 784 pixels into `28 × 28 × 1` format for CNN input.

Example:

```python
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0

X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)