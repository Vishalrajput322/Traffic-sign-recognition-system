from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model(r"F:\datasets\traffic sign dataset\training\TSR.h5")

classes = {
    0: "Speed limit (20Km/h)",
    1: "Speed limit (30Km/h)",
    2: "Speed limit (50Km/h)",
    3: "Speed limit (60Km/h)",
    4: "Speed limit (70Km/h)",
    5: "Speed limit (80Km/h)",
    6: "End of speed limit (80Km/h)",
    7: "Speed limit (100Km/h)",
    8: "Speed limit (120Km/h)",
    9: "No passing",
    10: "No passing vehicles over 3.5 tons",
    11: "Right-of-way at intersection",
    12: "Priority road",
    13: "Yield",
    14: "Stop",
    15: "No vehicles",
    16: "vehicles > 3.5 tons prohibited",
    17: "No entry",
    18: "General caution",
    19: "Dangerous curve left",
    20: "Dangerous curve right",
    21: "Double curve",
    22: "Bumpy road",
    23: "Slippery road",
    24: "Road narrows on the right",
    25: "Road work",
    26: "Traffic signals",
    27: "Pedestrians",
    28: "Children crossing",
    29: "Bicycles",
    30: "Beware of ice/snow",
    31: "Wild animals crossing",
    32: "End speed + passing limits",
    33: "Turn right ahead",
    34: "Turn left ahead",
    35: "Ahead only",
    36: "Go straight or right",
    37: "Go straight or left",
    38: "Keep right",
    39: "Keep left",
    40: "Round-about mandatory",
    41: "End of no passing",
    42: "End no passing vehicles > 3.5 tons"
}

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read the image.")
    img = cv2.resize(img, (30, 30))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    label = None

    if request.method == 'POST':
        if 'image' not in request.files:
            label = "No file part in the request"
            return render_template('index.html', prediction=prediction, label=label)

        file = request.files['image']

        if file.filename == '':
            label = "No file selected"
            return render_template('index.html', prediction=prediction, label=label)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            try:
                img = preprocess_image(filepath)
                pred = model.predict(img, verbose=0)
                class_id = int(np.argmax(pred))
                confidence = float(np.max(pred))

                label = f"{classes[class_id]} ({confidence:.2f})"
                prediction = filename  # Only pass filename to HTML

            except Exception as e:
                label = f"Prediction error: {str(e)}"
        else:
            label = "Unsupported file format. Allowed types: png, jpg, jpeg."

    return render_template('index.html', prediction=prediction, label=label)

if __name__ == '__main__':
    app.run(debug=True)
