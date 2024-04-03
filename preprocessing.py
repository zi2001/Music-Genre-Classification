import os
import librosa
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
import soundfile as sf

def load_audio_and_preprocess(filepath):
    y, sr = librosa.load(filepath)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    audio_features = np.concatenate((mel_spectrogram, mfcc), axis=0)
    audio_features = audio_features.astype(np.float32)
    return audio_features

def load_and_preprocess_data(data_path):
    audio_data = []
    genre_labels = []

    for genre_dir in os.listdir(data_path):
        genre_label = genre_dir
        subdirectory_path = os.path.join(data_path, genre_dir)

        for filename in os.listdir(subdirectory_path):
            if filename.endswith(".wav"):
                filepath = os.path.join(subdirectory_path, filename)
                audio_data.append(load_audio_and_preprocess(filepath))
                genre_labels.append(genre_label)

    audio_data_np = np.array(audio_data)
    genre_labels_np = np.array(genre_labels)

    encoder = LabelEncoder()
    integer_encoded_labels = encoder.fit_transform(genre_labels_np)
    y = to_categorical(integer_encoded_labels)

    scaler = StandardScaler()
    X = scaler.fit_transform(audio_data_np.reshape(-1, 141 * 216))

    return X, y
