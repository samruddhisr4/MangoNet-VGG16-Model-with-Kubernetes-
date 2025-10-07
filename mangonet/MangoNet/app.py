from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load your trained model (replace with your model path)
# NOTE: Ensure the model file `mango_classification_model.h5` is present in the container context or adjust MODEL_PATH
MODEL_PATH = os.environ.get('MODEL_PATH', 'mango_classification_model.h5')
import h5py

# Robust model loading: try to load at startup, but if it fails allow lazy loading
model = None

def load_model_safe():
    global model
    if model is not None:
        return model

    # Check file exists first
    if not os.path.exists(MODEL_PATH):
        app.logger.error('Model file not found at %s', MODEL_PATH)
        raise FileNotFoundError(f'Model file not found at {MODEL_PATH}')

    try:
        # Use compile=False to avoid issues if optimizer state or custom objects aren't present
        app.logger.info('Loading model from %s', MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        app.logger.info('Model loaded successfully')
        return model
    except Exception as e:
        app.logger.exception('Failed to load model: %s', e)
        # Try a fallback: rebuild the original architecture (VGG16 base + head) and load weights
        try:
            app.logger.info('Attempting fallback: rebuild architecture and load weights from HDF5')
            # infer number of classes from CLASS_NAMES if available
            num_classes = len(CLASS_NAMES) if 'CLASS_NAMES' in globals() and CLASS_NAMES else 8

            # build model same as in training notebook
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
            for layer in base_model.layers:
                layer.trainable = False

            fallback = Sequential()
            fallback.add(base_model)
            fallback.add(Flatten())
            fallback.add(Dense(256, activation='relu'))
            fallback.add(Dropout(0.5))
            fallback.add(Dense(num_classes, activation='softmax'))

            # compile the model (required before loading weights in some Keras versions)
            fallback.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

            # load weights from the HDF5 file
            fallback.load_weights(MODEL_PATH)
            model = fallback
            app.logger.info('Fallback model built and weights loaded successfully')
            return model
        except Exception as e2:
            app.logger.exception('Fallback weight-load also failed: %s', e2)
            # re-raise the original exception to signal failure to caller
            raise

# Try eager load at startup to fail fast in many deployments; if it fails the app will still run
try:
    load_model_safe()
except Exception:
    app.logger.warning('Model did not load at startup; it will be loaded on first prediction attempt')


# List of class names (replace with your actual classes)
CLASS_NAMES = ['Anwar Ratool', 'Chaunsa (Black)', 'Chaunsa (Summer Bahisht)', 'Chaunsa (White)', 'Dosehri', 'Fajri', 'Langra', 'Sindhri']


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/upload_page')
def upload_page():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('home'))
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        img = image.load_img(filepath, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0

        # Predict (ensure model is loaded)
        try:
            mdl = load_model_safe()
        except Exception as e:
            app.logger.error('Prediction failed because model could not be loaded: %s', e)
            # Show a friendly error page or redirect
            return render_template('results.html', prediction='Model not available', image_url=filepath)

        prediction = mdl.predict(img_array)
        predicted_class = CLASS_NAMES[np.argmax(prediction)]

        return render_template('results.html', prediction=predicted_class, image_url=filepath)


@app.route('/health')
def health():
    # simple health endpoint for container healthchecks
    return 'ok', 200


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    # bind to 0.0.0.0 so the container exposes the port
    app.run(host='0.0.0.0', port=port, debug=False)
