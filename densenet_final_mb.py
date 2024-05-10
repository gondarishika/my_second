# -*- coding: utf-8 -*-
"""
Created on Thu Feb 29 23:46:20 2024

@author: rishika
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Set the paths to your train and test data folders
train_data_dir = "C:/Users/rishika/Desktop/Liver Project/Images/train"
test_data_dir = "C:/Users/rishika/Desktop/Liver Project/Images/test"

# Set the image dimensions and batch size
img_width, img_height = 224, 224
batch_size = 24

# Set the number of classes
num_classes = 3

# Create data generators for train and test sets with data augmentation for the train set
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load the DenseNet121 model without the top classification layer
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze all layers in the base model
for layer in base_model.layers:
    layer.trainable = False

# Add a new classification layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=20
)

# Evaluate the model on train and test sets
train_loss, train_accuracy = model.evaluate(train_generator)
test_loss, test_accuracy = model.evaluate(test_generator)

print(f'Train Accuracy: {train_accuracy}')
print(f'Test Accuracy: {test_accuracy}')
# Save the trained model
model.save('densenet_balancedata.h5')

# Load the saved model for prediction
model = tf.keras.models.load_model('densenet_balancedata.h5')

# Define the target image size
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np
# Define the target image size
target_size = (224, 224)

# Load and preprocess the image you want to make predictions on
img_path = "C:/Users/rishika/Desktop/Liver Project/Images/test/Normal Liver/LeicaWebViewerSnapshot - 2023-12-30T114326.879.jpg"
img = load_img(img_path, target_size=target_size)  # Resize the image
img_array = img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.

# Make predictions
predictions = model.predict(img_array)

# Decode the predictions
predicted_classes = np.argmax(predictions, axis=1)

# Define class labels (replace with your own class labels)
class_labels = ['Cholangiocarcinoma', 'HCC', 'Normal_Liver']

# Print the predicted class
predicted_class_label = class_labels[predicted_classes[0]]
print("Predicted class:", predicted_class_label)

