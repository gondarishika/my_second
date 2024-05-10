import tensorflow as tf
from tensorflow.keras.applications import NASNetLarge
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

# Define directories
train_data_dir = "C:/Users/rishika/Desktop/Liver Project/Client_Data"
valid_data_dir = "C:/Users/rishika/Desktop/Liver Project/Client_Data"

# Image dimensions
img_width, img_height = 331, 331
batch_size = 16
epochs = 5
num_classes = 3  # Number of classes

# Load NASNetLarge model pre-trained on ImageNet
base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))

# Freeze all layers except for the last one
for layer in base_model.layers[:-2]:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load and preprocess training data with balanced class distribution
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True  # Shuffle the data to ensure randomness
)

# Load and preprocess validation data
valid_generator = test_datagen.flow_from_directory(
    valid_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical'
)

# Compute class weights for balancing
class_weights = {}
total_samples = np.zeros(num_classes)

# Count the number of samples in each class
for i in range(num_classes):
    total_samples[i] = np.sum(train_generator.classes == i)

# Compute class weights based on the total number of samples
for i in range(num_classes):
    class_weights[i] = total_samples.sum() / (num_classes * total_samples[i])

print("Class Weights:", class_weights)

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.samples // batch_size,
    class_weight=class_weights
)

# Evaluate the model on the validation set
test_loss, test_accuracy = model.evaluate(valid_generator)
print("Validation Loss:", test_loss)
print("Validation Accuracy:", test_accuracy)

# Save the trained model
model.save('nasnet_model_with_balanced_data.h5')

# Load the saved model for prediction
model = tf.keras.models.load_model('nasnet_model_with_balanced_data.h5')

# Define the target image size
img_width, img_height = 331, 331

# Load and preprocess the image you want to make predictions on
img_path =  "C:/Users/rishika/Desktop/Liver Project/Client_Data/Normal_Liver/Copy of LeicaWebViewerSnapshot - 2024-01-02T132412.135.jpg"
img = load_img(img_path, target_size=(img_width, img_height))
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
