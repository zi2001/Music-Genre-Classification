# TODO: Import your package, replace this by explicit imports of what you need

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import librosa
import io
import numpy as np
from tensorflow.keras.models import load_model

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

    y, sr = librosa.load(io.BytesIO(contents), sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mel_spectrogram = mel_spectrogram.astype(np.float32)

  # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

  # Chroma features
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=12)

  # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)

  # Combine features
    features = np.vstack([mel_spectrogram, mfccs, chroma_stft, spectral_centroid])
    print(features.shape)
    features = np.expand_dims(features,axis=0)
    print(features.shape)

    prediction = model.predict([features])
    index = list(np.argsort(prediction)[0][-1:])
    print(index)
    print(prediction)
    print(type(index))



    return { "prediction": float(index[0]) }

#return { "prediction":GENRES[float(index[0])] }
