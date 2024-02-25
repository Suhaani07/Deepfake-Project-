import streamlit as st
import numpy as np
import librosa
from keras.models import model_from_json
from keras.models import load_model
from PIL import Image
from keras.preprocessing import image
import cv2

# Function to load the deepfake detection model
def load_deepfake_model():
    json_file = open('model_architecture.json','r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    model.load_weights('cnn97.h5')
    return model

def load_deepfake_modelimage():
    model = load_model('DeepfakeImage.h5')  # Assuming the model is in the same directory
    return model

def detect_image_deepfake(image_file):
    # Load the model
    model = load_deepfake_modelimage()

    # Preprocess the image
    new_image_path = image_file


# Load the image and preprocess it for the model
    img = image.load_img(new_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.  # Normalize the pixel values

# Make predictions using the loaded model
    predictions = model.predict(img_array)

# Assuming binary classification (real vs. fake)
# You may have to adjust this based on your model's output
    if predictions[0, 0] > 0.5:
        print("Prediction: Real Image")
    else:
        print("Prediction: Fake Image")
        
    print(predictions[0,0])
        
    if predictions[0,0] < 0.5:
        st.write("Prediction: Fake", unsafe_allow_html=True)
    else:
        st.write("Prediction: Real", unsafe_allow_html=True)

        
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
    print(prediction)
    if prediction < 0.7:
        st.write("Prediction: Fake", unsafe_allow_html=True)
    else:
        st.write("Prediction: Real", unsafe_allow_html=True)





def calculate_ssim(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    (score, _) = cv2.compare_ssim(gray1, gray2, full=True)
    return score

# Function to check if the video is real or fake
def check_video_real_fake(video_file):
    # Convert video file uploader to OpenCV VideoCapture object
    video_bytes = video_file.read()
    video_array = np.frombuffer(video_bytes, dtype=np.uint8)

    try:
        # Initialize variables
        cap = None
        reference_frame = None
        total_frames = 0
        similar_frames = 0

        # Decode video and set properties
        cap = cv2.VideoCapture()
        if cap.open(cv2.CAP_FFMPEG):
            if cap.grab():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                ret, reference_frame = cap.retrieve()

                # Variables to count similar frames
                similar_frames = 0

                # Loop through each frame
                for _ in range(total_frames):
                    ret, frame = cap.read()

                    # Break the loop if no frame is read
                    if not ret:
                        st.write("Error: Unable to read frames from the video.")
                        break

                    # Calculate structural similarity
                    ssim_score = calculate_ssim(reference_frame, frame)

                    # Check if frames are more than 90% similar
                    if ssim_score >= 0.90:
                        similar_frames += 1

                    # Update reference_frame for the next iteration
                    reference_frame = frame

        # Check if there are frames before calculating similarity percentage
        if total_frames > 0:
            # Calculate similarity percentage
            similarity_percentage = (similar_frames / total_frames) * 100

            # Determine if the video is real or fake based on similarity percentage
            if similarity_percentage >= 90:
                st.write(f"Similarity Percentage: {similarity_percentage:.2f}%")
                st.write("Video is Real")
            else:
                st.write(f"Similarity Percentage: {similarity_percentage:.2f}%")
                st.write("Video is Fake")
        else:
            st.write("Video is Fake")

    except Exception as e:
        st.write(f"Error: {str(e)}")

    finally:
        # Release video capture object if it was created
        if cap is not None:
            cap.release()




def main():
    st.title("Deepfake Detector")

    # Button to select file type
    file_type = st.radio("Select file type", ["Image", "Video", "Audio"])

    # Upload file based on selected file type
    if file_type == "Image":
        st.write("Upload Image")
        image_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            st.image(image_file, caption="Uploaded Image (50x50)", width=50, use_column_width=False)
        detect_image_deepfake(image_file)
    elif file_type == "Video":
        st.write("Upload Video")
        video_file = st.file_uploader("Upload Video", type=["mp4"])
        if video_file is not None:
            st.video(video_file)
            check_video_real_fake(video_file)
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
