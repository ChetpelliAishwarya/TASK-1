import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import joblib
import librosa
import numpy as np
from keras.models import load_model
import tempfile
import os
import audioread

def extract_features(file_name, file_type):
    if file_type == "wav":
        y, sr = librosa.load(file_name, sr=None)
    elif file_type == "mp3":
        y, sr = librosa.load(file_name, sr=None, mono=True)
    else:
        raise ValueError(f"Unsupported file type: {file_type}. Only .wav and .mp3 are supported.")
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    meanfreq = np.mean(spectral_centroid)
    sd = np.std(spectral_centroid)
    median = np.median(spectral_centroid)
    Q25 = np.percentile(spectral_centroid, 25)
    Q75 = np.percentile(spectral_centroid, 75)
    IQR = Q75 - Q25
    
    n_bands = 6  
    fmin = 20.0  
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=n_bands, fmin=fmin)
    skew = np.mean(spectral_contrast)
    kurt = np.var(spectral_contrast)
    
    sp_ent = np.mean(librosa.feature.spectral_flatness(y=y))
    sfm = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    mode = np.mean(librosa.feature.zero_crossing_rate(y=y))
    centroid = meanfreq 
    harmonic = librosa.effects.harmonic(y)
    meanfun = np.mean(harmonic)
    minfun = np.min(harmonic)
    maxfun = np.max(harmonic)
    
    delta = librosa.feature.delta(y)
    meandom = np.mean(delta)
    mindom = np.min(delta)
    maxdom = np.max(delta)
    dfrange = maxdom - mindom
    modindx = mode  
    features = [meanfreq, sd, median, Q25, Q75, IQR, skew, kurt, sp_ent, sfm,
                mode, centroid, meanfun, minfun, maxfun, meandom, mindom,
                maxdom, dfrange, modindx]
    
    return np.array(features)

def extract_mfcc(filename, file_type, duration=3, offset=0.5):
    try:
        if file_type == "wav":
            y, sr = librosa.load(filename, duration=duration, offset=offset)
        elif file_type == "mp3":
            y, sr = librosa.load(filename, sr=None, mono=True, duration=duration, offset=offset)
        else:
            raise ValueError(f"Unsupported file type: {file_type}. Only .wav and .mp3 are supported.")
        
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        return mfcc
    except Exception as e:
        st.error(f"Error extracting MFCC features: {e}")
        return None

Gender_model_path = r'C:\Users\chais\Downloads\TASK-1\Gender prediction\Gender prediction model.pkl'
Scaler_path = r'C:\Users\chais\Downloads\TASK-1\Gender prediction\Scaler.pkl'
gender_model = joblib.load(Gender_model_path)
scaler = joblib.load(Scaler_path)

Emotion_model_path = r'C:\Users\chais\Downloads\TASK-1\emotion detection\emotion detection from voice.h5'
emotion_model = load_model(Emotion_model_path)

st.title('Gender and Emotion Recognition from .wav or .mp3 Files')

uploaded_file = st.file_uploader("Choose a .wav or .mp3 file", type=["wav", "mp3"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')
    
    file_type = uploaded_file.name.split(".")[-1]
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_file.read())
    
    with st.spinner('Predicting Gender...'):
        input_features = extract_features(temp_file.name, file_type)
        input_features = scaler.transform(input_features.reshape(1, -1))
        predicted_gender = gender_model.predict(input_features)
    
    predicted_gender_label = "Male" if predicted_gender == 0 else "Female"
    st.write("Predicted Gender:", predicted_gender_label)

    if predicted_gender_label == "Female":
        with st.spinner('Predicting Emotion...'):
            mfcc_features = extract_mfcc(temp_file.name, file_type)

        if mfcc_features is not None:
            X_new = np.expand_dims(mfcc_features, axis=0)
            X_new = np.expand_dims(X_new, axis=-1)
            
            emotion_label = emotion_model.predict(X_new)
            emotion_label = np.argmax(emotion_label, axis=1)
            
            emotion_mapping = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'ps', 6: 'sad'}
            predicted_emotion = emotion_mapping.get(emotion_label[0])
            st.write("Predicted Emotion:", predicted_emotion)
        else:
            st.write("Failed to extract MFCC features. Please try another file.")
    else:
        st.write("Please upload a female voice for emotion prediction.")
    
    os.remove(temp_file.name)
