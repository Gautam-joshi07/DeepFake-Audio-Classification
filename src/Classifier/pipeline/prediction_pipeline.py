import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
# from tensorflow.keras.models import load_model
from keras import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import streamlit as st

class AudioPredictionPipeline:
    def __init__(self, file_sound):
        self.file_sound = file_sound
        self.audio_file_path = os.path.join('audio_files', self.file_sound.name)
        self.spectrogram_path = 'mel_spectrogram.png'
        self.model = load_model(os.path.join('model', 'model.h5'))
    
    def file_save(self):
        with open(self.audio_file_path, 'wb') as f:
            f.write(self.file_sound.getbuffer())
        return self.audio_file_path

    def create_spec(self):
        y, sr = librosa.load(self.audio_file_path)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(mel, ref=np.max)
        
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.savefig(self.spectrogram_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        image_data = load_img(self.spectrogram_path, target_size=(224, 224))
        return image_data

    def pred(image_data, model):
        img_array = np.array(image_data)
        img_array1 = img_array / 255
        img_batch = np.expand_dims(img_array1, axis=0)

        prediction = model.predict(img_batch)
        class_label = np.argmax(prediction)

        return class_label, prediction

# Example of how to use the class

# pipeline = AudioPredictionPipeline(file_sound)
# saved_file = pipeline.file_save()
# image_data = pipeline.create_spec()
# result = pipeline.predict(image_data)

