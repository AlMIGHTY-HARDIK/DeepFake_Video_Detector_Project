# DeepFake Video Detector Project

This repository hosts the **DeepFake Video Detector Project**, a Flask-based web server that uses a pre-trained TensorFlow model to analyze video frames for deepfake content. Leveraging efficient batch processing and OpenCV for video handling, this application classifies uploaded videos as either "Real" or "Deepfake."

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Project Structure](#project-structure)
- [Contributing](#contributing)


## Overview

This web application:
- Accepts video files via an intuitive web interface.
- Processes video frames in batches to optimize prediction efficiency.
- Uses a deepfake detection model to classify frames, providing a clear prediction (Real/Deepfake) based on the analysis.
- Returns a detailed JSON response including the total frames analyzed, number of deepfake frames, and the ratio of deepfake content.

## Features

- **Deepfake Detection**: Employs a trained model to detect deepfake frames in video files.
- **Batch Processing**: Handles video frames in batches for optimized processing.
- **Intuitive API**: Easy-to-use API endpoint for uploading and classifying videos.
- **Efficient Preprocessing**: Applies normalization and resizing to frames, ensuring consistency and accuracy in model predictions.

## Installation

To run this project locally, please ensure you have the following prerequisites installed:

- **Python 3.7+**
- **TensorFlow** and **Keras**
- **OpenCV**

Follow these steps to set up and run the project:

1. **Clone the repository**
   ```bash
   git clone https://github.com/username/deepfake-detection-app.git
   cd deepfake-detection-app
2. Install dependencies

    ```bash
    Copy code
    pip install -r requirements.txt
    ```
3. Run the application

    ```bash
    
    python app.py
    ```
4. Access the web application
Open your browser and navigate to http://localhost:5000.

## Usage
1. Upload a video file using the provided interface.
2. The application processes the video in batches, detects deepfake frames, and calculates the deepfake ratio.
3. Results are returned in JSON format, containing:
- prediction: "Real" or "Deepfake"
- deepfake_ratio: The percentage of frames classified as deepfake
- total_frames and deepfake_frames: Counts of total and deepfake frames
## Example JSON Response
```json

{
  "prediction": "Deepfake",
  "deepfake_ratio": 0.8,
  "total_frames": 120,
  "deepfake_frames": 96
}
```
## API Reference
- GET /: Returns the HTML interface for video upload.
- POST /predict: Accepts a video file and returns a deepfake prediction.
  - Parameters:
     - file (form-data): The video file to be analyzed.
  - Response: JSON with prediction results (see example above).
## Model Details
The deepfake detection model is loaded with a TFSMLayer layer, utilizing TensorFlow Serving for efficient, production-grade inference. Each frame is preprocessed to ensure compatibility with the model, which uses a 224x224 input shape and RGB color space.

## Project Structure
```plaintext

deepfake-detection-app/
├── app.py               # Main application script
├── requirements.txt     # Project dependencies
├── templates/
│   └── index.html       # HTML template for web interface
├── model/
      └── saved_model.pb                  # Main model file
      └── variables/
          ├── variables.data-00000-of-00001 # Model weights data file
          └── variables.index               # Index file for model weights

└── README.md            # This README file
```
## Contributing
Contributions are welcome! If you have suggestions for improvement or find any issues, please open an issue or submit a pull request. When contributing, please follow the project's code of conduct.


