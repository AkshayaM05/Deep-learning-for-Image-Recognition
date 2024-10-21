from flask import Flask, request, render_template
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

# Load the MobileNetV2 model
model = MobileNetV2(weights='imagenet')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    if file:
        # Save the uploaded image
        img_path = os.path.join('static', file.filename)
        file.save(img_path)

        # Preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        # Make predictions
        predictions = model.predict(img_array)
        decoded_predictions = decode_predictions(predictions, top=1)[0]
        label = decoded_predictions[0][1]  # The second element in the tuple is the label
        
        return render_template('result.html', image_file=file.filename, label=label)

if __name__ == '__main__':
    app.run(debug=True)
