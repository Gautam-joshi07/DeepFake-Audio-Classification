import os
import numpy as np
from flask import Flask, request, render_template, jsonify, send_file
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import librosa
import librosa.display
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'audio_files'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3'}
app.config['STATIC_FOLDER'] = 'static'
class_names = ['real', 'fake']

# Load the model once, when the server starts
model = load_model(os.path.join("artifacts/training", "model.h5"))

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def file_save(file_sound):
    filename = secure_filename(file_sound.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_sound.save(filepath)
    return filepath

def create_spec(sound):
    audio_file = sound
    y, sr = librosa.load(audio_file)
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(mel, ref=np.max)
    
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    
    # Save the spectrogram image in the static folder
    spec_image_path = os.path.join(app.config['STATIC_FOLDER'], 'mel_spectrogram.png')
    plt.savefig(spec_image_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    image_data = load_img(spec_image_path, target_size=(224, 224))
    return spec_image_path, image_data

def pred(image_data, model):
    img_array = img_to_array(image_data)
    img_array = img_array / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_batch)
    class_label = np.argmax(prediction)

    return class_label, prediction

@app.route('/')
def upload_form():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filepath = file_save(file)
        spec_image_path, image_data = create_spec(filepath)
        class_label, prediction = pred(image_data, model)
        result = {"prediction": class_names[class_label], "image_path": spec_image_path}
        return jsonify(result)
    else:
        return jsonify({'error': 'File type not allowed'})

@app.route('/record', methods=['POST'])
def record_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio part'})
    file = request.files['audio']
    if file and allowed_file(file.filename):
        filepath = file_save(file)
        spec_image_path, image_data = create_spec(filepath)
        class_label, prediction = pred(image_data, model)
        result = {"prediction": class_names[class_label], "image_path": spec_image_path}
        return jsonify(result)
    else:
        return jsonify({'error': 'File type not allowed'})

if __name__ == "__main__":
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['STATIC_FOLDER']):
        os.makedirs(app.config['STATIC_FOLDER'])
    app.run(host="0.0.0.0")
