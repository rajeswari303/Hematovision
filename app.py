from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)
model = tf.keras.models.load_model("Blood_Cell.h5")

IMG_SIZE = 224

# Map label index to class name
class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    # Save uploaded image temporarily
    filepath = os.path.join("static", file.filename)
    file.save(filepath)

    # Read and preprocess
    img = cv2.imread(filepath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    predicted_class = class_labels[np.argmax(pred)]

    return render_template("result.html", prediction=predicted_class, image_file=file.filename)

if __name__ == "__main__":
    app.run(debug=True)