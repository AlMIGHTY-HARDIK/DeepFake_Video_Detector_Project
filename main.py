import tensorflow as tf
from keras.layers import TFSMLayer
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np
import os

# Load pre-trained model with TFSMLayer
model = TFSMLayer("model", call_endpoint='serving_default')

# Preprocess frame function optimized for batch processing
def preprocess_frames(frames):
    frames = np.array([cv2.resize(frame, (224, 224)) for frame in frames])
    frames = frames[..., ::-1]  # Convert BGR to RGB in a vectorized way
    frames = frames / 255.0
    frames = (frames - np.mean(frames, axis=(1, 2, 3), keepdims=True)) / (np.std(frames, axis=(1, 2, 3), keepdims=True) + 1e-8)
    return frames

# Flask app for interaction
app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file:
        video_path = "temp_video.mp4"
        file.save(video_path)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        deepfake_count = 0
        batch_size = 32  # Increase batch size for efficiency
        batch_frames = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            batch_frames.append(frame)

            if len(batch_frames) == batch_size:
                # Process batch frames
                batch_frames_np = preprocess_frames(batch_frames)
                predictions = model(batch_frames_np)

                if isinstance(predictions, dict):
                    predictions = list(predictions.values())[0]  # Extract the tensor value

                deepfake_count += int(np.sum(predictions.numpy() > 0.75))  # Convert to int for JSON compatibility
                batch_frames = []  # Clear batch

        # Process remaining frames
        if batch_frames:
            batch_frames_np = preprocess_frames(batch_frames)
            predictions = model(batch_frames_np)
            if isinstance(predictions, dict):
                predictions = list(predictions.values())[0]

            deepfake_count += int(np.sum(predictions.numpy() > 0.75))  # Convert to int for JSON compatibility

        cap.release()
        os.remove(video_path)

        # Calculate results
        deepfake_ratio = float(deepfake_count) / frame_count if frame_count else 0  # Convert to float for JSON compatibility
        result = "Deepfake" if deepfake_ratio > 0.7 else "Real"

        return jsonify({
            "prediction": result,
            "deepfake_ratio": deepfake_ratio,
            "total_frames": int(frame_count),   # Ensure frame_count is int for JSON
            "deepfake_frames": int(deepfake_count)  # Ensure deepfake_count is int for JSON
        })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
