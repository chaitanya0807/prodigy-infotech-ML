import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Step 1: Load the Dataset
# Specify the path to the dataset folder containing 'train' images
data_path = 'path_to_dataset/train'

# Step 2: Preprocess Images
X = []
y = []
img_size = 64  # Resize images to 64x64 pixels

# Load images and labels
for category in ['cat', 'dog']:
    folder_path = os.path.join(data_path, category)
    label = 0 if category == 'cat' else 1
    for img in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img)
        try:
            img_array = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
            resized_array = cv2.resize(img_array, (img_size, img_size))  # Resize image
            X.append(resized_array)
            y.append(label)
        except Exception as e:
            pass

# Convert to numpy arrays
X = np.array(X).reshape(-1, img_size * img_size)  # Flatten images
y = np.array(y)

# Step 3: Split into Training and Test Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Create and Train SVM Classifier
clf = svm.SVC(kernel='linear')  # Use linear kernel for SVM
clf.fit(X_train, y_train)

