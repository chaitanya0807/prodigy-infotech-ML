# Cat and Dog Image Classification using SVM

## Project Overview
This project implements an SVM (Support Vector Machine) model to classify images of cats and dogs. The images are preprocessed, converted to grayscale, and resized before feeding them into the SVM classifier. The goal is to distinguish between the two classes: cats and dogs.

## Dataset
The dataset used is the **"Dogs vs. Cats"** dataset from Kaggle, containing labeled images of cats and dogs stored in separate folders.

## Project Steps
1. **Load and Preprocess Images**:
   - Images are converted to grayscale.
   - Resized to 64x64 pixels.
   - Flattened into 1D arrays.

2. **Train-Test Split**:
   - Dataset is split into training and test sets using `train_test_split`.

3. **Model Training**:
   - Trained an SVM classifier using a linear kernel.

4. **Model Evaluation**:
   - Evaluated using accuracy score on the test set.

## Installation
1. Install required libraries:
   ```bash
   pip install numpy opencv-python scikit-learn
   ```

2. Download the dataset from [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats/data).

3. Run the script:
   ```bash
   python cat_dog_classification.py
   ```

## Results
The final accuracy of the model on the test set is printed after evaluation. The model can be improved with deeper feature extraction and different kernels.
