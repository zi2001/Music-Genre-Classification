# TODO: Import your package, replace this by explicit imports of what you need

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import librosa
import io
import numpy as np
from tensorflow.keras.models import load_model
from gmc.params import GENRES

app = FastAPI()


model = load_model('model/model_best.h5')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

@app.post('/upload_music')
async def receive_music(mus: UploadFile):
    ### Receiving and decoding the image
    contents = await mus.read()

    input_y, sr = librosa.load(io.BytesIO(contents), sr=None)



    y = input_y[0:0 + sr * 5]






    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    #mel_spectrogram = mel_spectrogram.astype(np.float32)
    # Extract Mel spectrogram


    # Extract MFCC features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Combine Mel spectrogram and MFCC features
    audio_features = np.concatenate((mel_spectrogram, mfcc), axis=0)
    audio_features = audio_features.astype(np.float32)  # Convert to float32 for CNN


    audio_features = np.expand_dims(audio_features,axis=0)


    prediction = model.predict([audio_features])
    index = list(np.argsort(prediction)[0][-1:])
    print(index)
    print(prediction)
    print(type(index))



    return { "prediction":GENRES[int(index[0])] }
