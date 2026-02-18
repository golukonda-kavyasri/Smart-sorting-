from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import uuid

app = Flask(__name__)

# Config
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Load model and class labels
MODEL_PATH = 'healthy_vs_rotten.h5'
LABELS_PATH = 'class_labels.json'

model = None
class_labels = []

def load_resources():
    global model, class_labels
    try:
        model = load_model(MODEL_PATH)
        print('✅ Model loaded successfully!')
    except Exception as e:
        print(f'❌ Error loading model: {e}')

    try:
        with open(LABELS_PATH, 'r') as f:
            class_labels = json.load(f)
        print(f'✅ Class labels loaded: {len(class_labels)} classes')
    except Exception as e:
        print(f'❌ Error loading class labels: {e}')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict(img_path):
    img = load_img(img_path, target_size=(224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_index = np.argmax(predictions)
    predicted_class = class_labels[predicted_index]
    confidence = float(np.max(predictions)) * 100

    # Determine condition (fresh or rotten)
    condition = 'Fresh' if 'fresh' in predicted_class.lower() else 'Rotten'

    # Get top 3 predictions
    top3_indices = np.argsort(predictions)[::-1][:3]
    top3 = [
        {
            'label': class_labels[i],
            'confidence': round(float(predictions[i]) * 100, 2)
        }
        for i in top3_indices
    ]

    return {
        'predicted_class': predicted_class,
        'condition': condition,
        'confidence': round(confidence, 2),
        'top3': top3
    }


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Use PNG, JPG, JPEG, or WEBP'}), 400

    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500

    # Save file with unique name
    filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    file.save(filepath)

    try:
        result = predict(filepath)
        result['image_url'] = f'/static/uploads/{filename}'
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/about')
def about():
    return render_template('about.html')


if __name__ == '__main__':
    load_resources()

    app.run(host='0.0.0.0', port=5000)
