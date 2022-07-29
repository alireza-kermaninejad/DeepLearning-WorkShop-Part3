from __future__ import division, print_function
import os
import numpy as np
import json
import requests
from PIL import Image

# Keras
# from keras.models import load_model
# from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename


# Model saved with Keras model.save()
# MODEL_PATH = 'C:/Users/Peyton/Desktop/vehicle_color_detection/app/vcd_models'
# Load your trained model
# model = load_model(MODEL_PATH)
# model.make_predict_function()   # Necessary
# print('Model loaded. Start serving...')
# print('Model loaded. Check http://127.0.0.1:5000/')

print('Model loaded. Check http://localhost:5000/')

classes = ["Red", "Gray", "Navy-blue", "White", "Shell-white", "Blue", "Brown", "Black", "Silver"]
MODEL_URI = 'http://localhost:8501/v1/models/vcd_models:predict'


def predict(filename):
    # img = image.load_img(filename, target_size=(112, 112))
    img = Image.open(filename).resize((112, 112))
    # img = image.img_to_array(img)
    img = np.array(img)
    img = np.expand_dims(img, axis=0)

    data = json.dumps({"signature_name": "serving_default",
                        "instances": img.tolist()})
    headers = {"content-type": "application/json"}
    json_response = requests.post(MODEL_URI, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions']
    class_predictions = classes[np.argmax(predictions[0])]
    
    print ('Prediction: '+str(class_predictions))
    return str(class_predictions)


# Define a flask app
app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/help', methods=['GET'])
def help_page():
    return render_template('help-index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        
        # Get final result
        answer = predict(file_path)
        
        return answer
    return None


# ///////////////////////////////////////////////////////////
# Errors Handling (Some HTTP status codes)
# 4xx client errors
@app.errorhandler(404)
def error_404(error):
    return render_template('errors/404.html'), 404

@app.errorhandler(400)
def error_400(error):
    return render_template('errors/400.html'), 400

@app.errorhandler(403)
def error_403(error):
    return render_template('errors/403.html'), 403

@app.errorhandler(500)
def error_500(error):
    return render_template('errors/500.html'), 500

@app.errorhandler(415)
def error_415(error):
    return render_template('errors/415.html'), 415
# ///////////////////////////////////////////////////////////


if __name__ == '__main__':
    app.run(debug=True)