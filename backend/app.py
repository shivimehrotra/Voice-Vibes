from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
from tensorflow import keras


import numpy as np
import io
import librosa
import base64

app = Flask(__name__)

# Load your CNN model
model = keras.model.load_model('C:\Users\shivi\Downloads\emotion-recognition.hdf5')

# Function to extract audio features from base64 audio data
def extract_features(audio_data):
    # Convert base64 audio data to bytes
    audio_bytes = base64.b64decode(audio_data)
    # Convert bytes to numpy array
    audio_array, _ = librosa.load(io.BytesIO(audio_bytes), sr=22050)
    # Extract audio features using librosa
    features = librosa.feature.mfcc(y=audio_array, sr=22050, n_mfcc=13)
    # Normalize features
    features = (features - np.mean(features)) / np.std(features)
    return features

def process_prediction(prediction):
    emotion_labels = ['Angry', 'Happy', 'Sad', 'Neutral']
    emotion_index = np.argmax(prediction)
    return emotion_labels[emotion_index]
# Endpoint for handling audio upload and prediction
@app.route('/predict', methods=['POST'])
def predict_emotion():
    try:
        # Get audio data from request
        audio_data = request.json['audioData']
        # Extract features from audio data
        features = extract_features(audio_data)
        # Reshape the features for prediction
        features = np.expand_dims(features, axis=0)
        # Make prediction using the loaded model
        prediction = model.predict(features)
        # Process the prediction as needed (e.g., convert to emotion label)
        emotion_label = process_prediction(prediction)
        # Return the predicted emotion label
        return jsonify({'emotion': emotion_label})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, request, jsonify
# from tensorflow.keras.models import load_model
# import numpy as np
# import base64

# app = Flask(__name__)

# # Load your CNN model
# model = load_model('path_to_your_model.hdf5')

# # Preprocess audio data (e.g., convert to spectrogram)
# def preprocess_audio(audio_data):
#     # Your preprocessing code here
#     return preprocessed_data

# # Endpoint for handling audio upload and prediction
# @app.route('/upload', methods=['POST'])
# def upload_file():
#     try:
#         # Get audio data from request
#         audio_data = request.json['audioData']
#         # Convert base64 audio data to bytes
#         audio_bytes = base64.b64decode(audio_data)
#         # Preprocess audio data
#         preprocessed_data = preprocess_audio(audio_bytes)
#         # Reshape the data if needed
#         preprocessed_data = np.expand_dims(preprocessed_data, axis=0)
#         # Make prediction using the loaded model
#         prediction = model.predict(preprocessed_data)
#         # Process the prediction as needed
#         # For example, convert prediction to emotion label
#         emotion_label = process_prediction(prediction)
#         # Return the predicted emotion label
#         return jsonify({'emotion': emotion_label})
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)

# from flask import Flask, request, jsonify
# from flask_cors import CORS 
# import random

# app = Flask(__name__)
# CORS(app)
# @app.route('/upload', methods=['POST'])
# def upload_audio():
#     try:
#         # Get the JSON data from the request   
#         print("hi") 
#         data = request.get_json()
#         audio_data = data.get('audioData')
#         # print(audio_data)
        
#         emotions = ['happy', 'sad', 'angry', 'surprised', 'excited', 'content', 'scared', 'disgusted', 'bored', 'relaxed']
#         random_emotion = random.choice(emotions)
#         return jsonify({'message': random_emotion})
        

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# if __name__ == '__main__':
#     app.run(debug=True)
