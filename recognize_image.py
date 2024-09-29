import numpy as np
import pandas as pd
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Step 1: Load the dataset
data_path = 'path_to_food-101'  
categories = os.listdir(data_path)
num_classes = len(categories)

# Step 2: Prepare data
def load_data():
    X = []
    y = []
    for label, category in enumerate(categories):
        category_path = os.path.join(data_path, category)
        for img in os.listdir(category_path):
            img_path = os.path.join(category_path, img)
            img_array = cv2.imread(img_path)
            img_array = cv2.resize(img_array, (128, 128))  # Resize to 128x128
            X.append(img_array)
            y.append(label)
    return np.array(X), np.array(y)

X, y = load_data()
X = X / 255.0  # Normalize pixel values

# Step 3: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Data Augmentation
datagen = ImageDataGenerator(horizontal_flip=True, rotation_range=20)
datagen.fit(X_train)

# Step 5: Build the CNN model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Step 6: Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Step 7: Train the model
model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))

# Step 8: Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Step 9: Predict calorie content (You can create a mapping based on categories)
def predict_calorie(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return categories[predicted_class]  # You can map class to calorie content

# Example usage
# image_path = 'path_to_test_image.jpg'
# food_item = predict_calorie(image_path)
# print(f'Predicted food item: {food_item}')
