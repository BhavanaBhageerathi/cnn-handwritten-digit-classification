# Handwritten Digit Classification using CNN

## Overview

This project implements a Convolutional Neural Network (CNN) using TensorFlow/Keras to classify handwritten digits from the MNIST dataset.

The model is trained to recognize handwritten digits from 0 to 9 using grayscale images of size 28 × 28 pixels.

It also demonstrates how to save and load a trained model and use it to predict digits from custom handwritten images.

## Features

* Loaded and explored the MNIST dataset.
* Preprocessed images by reshaping and normalizing pixel values.
* Applied one-hot encoding to target labels.
* Built a CNN using convolutional, max-pooling, dropout, flatten, and dense layers.
* Trained the model using the Adam optimizer.
* Evaluated model performance using test data.
* Saved the model architecture in JSON format.
* Saved trained model weights in HDF5 format.
* Loaded the saved model for prediction.
* Predicted handwritten digits from custom images.

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib
* Pillow
* MNIST Dataset

## Dataset

The project uses the MNIST dataset, which contains 70,000 grayscale handwritten digit images:

* 60,000 training images
* 10,000 testing images
* Image dimensions: 28 × 28 pixels
* Classes: Digits 0–9

The dataset is loaded directly using the Keras MNIST dataset utility.

## Model Architecture

The CNN consists of the following layers:

1. **Convolutional Layer:** Extracts important image features using 32 filters of size 5 × 5.
2. **Max Pooling Layer:** Reduces the spatial dimensions of feature maps.
3. **Dropout Layer:** Helps reduce overfitting.
4. **Flatten Layer:** Converts feature maps into a one-dimensional vector.
5. **Dense Layer:** Learns patterns for classification using 128 neurons.
6. **Output Layer:** Uses 10 neurons with Softmax activation to classify digits from 0 to 9.

## Model Configuration

* Input shape: 28 × 28 × 1
* Number of classes: 10
* Batch size: 200
* Epochs: 10
* Optimizer: Adam
* Loss function: Categorical Crossentropy
* Evaluation metric: Accuracy

## Model Saving and Loading

The trained model is saved in two parts:

* Model architecture: JSON file
* Model weights: HDF5 file

These files are used to reconstruct the trained model and perform predictions without retraining.

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/your-username/cnn-handwritten-digit-classification.git
```

### 2. Navigate to the project folder

```bash
cd cnn-handwritten-digit-classification
```

### 3. Create a virtual environment

```bash
python -m venv venv
```

### 4. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

### 6. Run the project

```bash
python cnn_mnist.py
```

The model will train on the MNIST dataset, evaluate its performance, and demonstrate digit prediction using a custom image.

## Project Output

The project demonstrates handwritten digit classification using a trained CNN model. It also predicts the digit represented in a custom input image.

## Author

Bhavana Bhageerathi
