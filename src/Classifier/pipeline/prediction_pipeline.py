import numpy as np
# from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

import os
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
# from tensorflow.keras.models import load_model
# from keras import load_model
# from tensorflow.keras.preprocessing.image import load_img, img_to_array



import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def create_spectrogram(self):
        y, sr = librosa.load(self.filename)
        mel = librosa.feature.melspectrogram(y=y, sr=sr)
        log_ms = librosa.power_to_db(mel, ref=np.max)

        plt.figure(figsize=(10, 4))
        librosa.display.specshow(log_ms, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel spectrogram')
        plt.tight_layout()
        plt.savefig('mel_spectrogram.png')
        plt.close()

    def predict(self):
        self.create_spectrogram()
        
        model = load_model(os.path.join("model", "model.h5"))
        image_path = 'mel_spectrogram.png'
        test_image = load_img(image_path, target_size=(224, 224))
        test_image = img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0
        
        result = np.argmax(model.predict(test_image), axis=1)
        
        if result[0] == 1:
            prediction = 'Real'
        else:
            prediction = 'Fake'
        
        return [{"prediction": prediction}]



