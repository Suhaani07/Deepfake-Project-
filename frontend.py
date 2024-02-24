import streamlit as st
import numpy as np
import librosa
from keras.models import model_from_json

# Function to load the deepfake detection model
def load_deepfake_model():
    json_file = open('model_architecture.json','r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights('cnn97.h5')
    return model

# Function to detect deepfake for audio
def detect_audio_deepfake(audio_file):
    # Load the model
    model = load_deepfake_model()

    # Load the audio file
    y, sr = librosa.load(audio_file, sr=None)
    y_1s = y[:sr] 

    # Calculate audio features
    chroma_stft = librosa.feature.chroma_stft(y=y_1s, sr=sr).mean()
    rms = librosa.feature.rms(y=y_1s).mean()
    spectral_centroid = librosa.feature.spectral_centroid(y=y_1s, sr=sr).mean()
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y_1s, sr=sr).mean()
    rolloff = librosa.feature.spectral_rolloff(y=y_1s, sr=sr).mean()
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y_1s).mean()
    mfccs = librosa.feature.mfcc(y=y_1s, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfccs, axis=1)

    # Concatenate features
    features = np.hstack([chroma_stft, rms, spectral_centroid, spectral_bandwidth, rolloff, zero_crossing_rate, mfcc_means])

    # Reshape features for prediction
    features = features.reshape(1, -1)

    # Make prediction
    prediction = model.predict(features)

    # Output prediction
    if prediction < 0.5:
        st.write("Prediction: Fake", unsafe_allow_html=True)
    else:
        st.write("Prediction: Real", unsafe_allow_html=True)

def main():
    st.title("Deepfake Detector")

    # Button to select file type
    file_type = st.radio("Select file type", ["Image", "Video", "Audio", "Message"])

    # Upload file based on selected file type
    if file_type == "Image":
        st.write("Upload Image")
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            st.image(image_file, caption="Uploaded Image", use_column_width=True)
    elif file_type == "Video":
        st.write("Upload Video")
        video_file = st.file_uploader("Upload Video", type=["mp4"])
        if video_file is not None:
            st.video(video_file)
    elif file_type == "Audio":
        st.write("Upload Audio")
        audio_file = st.file_uploader("Upload Audio", type=["mp3"])
        if audio_file is not None:
            detect_audio_deepfake(audio_file)
    elif file_type == "Message":
        st.write("Display Message")
        # Implement message logic here

if __name__ == "__main__":
    main()
