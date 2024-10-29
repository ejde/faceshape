# Face Shape Detector - Streamlit App

This Streamlit app allows users to determine their face shape by accessing their camera, taking a picture, and analyzing the facial landmarks. The app uses open-source libraries such as OpenCV, Dlib, and NumPy to detect facial landmarks and compute measurements that classify the face shape into categories like Oval, Round, Square, etc.

## Features

- Accesses the user's webcam for live capture.
- Detects facial landmarks to calculate facial proportions.
- Classifies the face shape based on key facial measurements (e.g., face length, cheekbone width, jawline).
- Displays reasons why the detected shape was chosen.
- Provides visual overlay of facial landmarks to guide the user.
  
## Requirements

- Python 3.8 or higher
- The following Python libraries:
  - `streamlit`
  - `opencv-python`
  - `dlib`
  - `face_recognition`
  - `imutils`
  - `numpy`

You can install all the required libraries using the following command:

```bash
pip install streamlit opencv-python dlib face-recognition imutils numpy

