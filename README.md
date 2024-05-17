
# Music-Genre-Classification

## Overview
This project aims to classify music genres using a Convolutional Neural Network (CNN) model. The model is trained on audio features extracted from music tracks, specifically spectrograms and Mel-Frequency Cepstral Coefficients (MFCC). The project leverages advanced machine learning techniques to achieve high accuracy in genre classification.

## Features
- **Spectrograms:** Visual representations of the spectrum of frequencies in a sound sample.
- **MFCC (Mel-Frequency Cepstral Coefficients):** Coefficients that collectively make up an MFC, representing the short-term power spectrum of a sound.

## Model
- **Convolutional Neural Network (CNN):** A deep learning model used for image recognition tasks, adapted here for audio signal processing.

## Tools and Technologies
- **Python:** Programming language used for developing the model.
- **TensorFlow and Keras:** Libraries used for building and training the CNN.
- **Librosa:** Python package for music and audio analysis, used for feature extraction.
- **Google Cloud Platform (GCP):** Initially used for model deployment, now decommissioned due to cost considerations.

## Project Structure
The project is organized as follows:
Music-Genre-Classification/
├── data/ # Directory containing the dataset
├── src/ # Directory containing the source code
│ ├── data_preprocessing.py # Script for extracting features (spectrograms and MFCC) from audio files
│ ├── model.py # Script for building and training the CNN model
│ ├── train.py # Script for training the model
│ ├── evaluate.py # Script for evaluating the model
├── notebooks/ # Jupyter notebooks for exploratory data analysis and model experimentation
├── requirements.txt # List of dependencies required to run the project
├── README.md # Project README file
└── deploy/ # Directory containing deployment instructions and code for GCP


## Deployment
Note: The Google Cloud Platform (GCP) deployment has been decommissioned due to cost considerations. 

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for any bugs or feature requests.

## Authors
Kyrillos Tadros
Zineb Hilary Hamdi

## Acknowledgements
This project was built during the Data Science and AI bootcamp at Le Wagon.

Feel free to contact us for any further information or questions regarding the project.
