from flask import Flask, request, render_template, redirect, url_for
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the saved model
model = tf.keras.models.load_model('tools_classification.keras')


# Define the labels for medicinal and toxic plants
labels = [
    "Gasoline Can", "Hammers", "Pebbles","Rope","Screw Driver","Toolbox","Wrench","pliers",
]

# Function to preprocess an image
def preprocess_image(img_path, target_size=(224, 224)):
    img = image.load_img(img_path, target_size=target_size)  # Load and resize image
    img_array = image.img_to_array(img)  # Convert to numpy array
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions to create a batch-like effect
    img_array = img_array / 255.0  # Normalize the image
    return img_array

# Function to predict and get the category
def predict_category(img_path):
    img_array = preprocess_image(img_path)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the predicted class index
    predicted_class = np.argmax(predictions, axis=1)[0]
     
    print(f'Predicted class index: {predicted_class}')
   
    return (f"The image is classified as : {labels[predicted_class]}")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            prediction = predict_category(file_path)
            os.remove(file_path)  # Remove the file after prediction
            return render_template('result.html', prediction=prediction)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)