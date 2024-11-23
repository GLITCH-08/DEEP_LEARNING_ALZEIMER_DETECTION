from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Initialize Flask app
app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model
model = load_model('alzheimer_vgg16_40.h5')

# Class labels (modify as per your dataset)
class_labels = ['Mild Impairment', 'Moderate Impairment', 'Non Impairment', 'Very Mild Impairment']

# Helper function to check file extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Route for home page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img = load_img(file_path, target_size=(150, 150))  # Resize as per your model
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = class_labels[np.argmax(predictions)]

        return render_template('index.html', prediction=predicted_class, image_url=file_path)

    return redirect(request.url)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
