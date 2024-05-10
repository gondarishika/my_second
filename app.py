from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.densenet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

app = Flask(__name__)

# Load the saved model
model = load_model(r"C:\Users\rishika\Desktop\Liver Project\New folder (2)\densenet_balancedata.h5")
model.trainable = True
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image file
    file = request.files['file']
    img = Image.open(file)
    img = img.resize((224, 224))  # Resize the image to match the model's input size

    # Convert the image to RGB format
    img = img.convert('RGB')

    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)

    # Make a prediction
    prediction = model.predict(np.expand_dims(img_array, axis=0))
    class_names = ['Cholangiocarcinoma', 'HCC', 'Normal_Liver']
    predicted_class_index = np.argmax(prediction)
    predicted_class = class_names[predicted_class_index]

    return render_template('index.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=False)