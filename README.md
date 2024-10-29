# Face Shape Detector - Streamlit App

This Streamlit app allows users to determine their face shape by accessing their camera, taking a picture, and analyzing the facial landmarks. The app uses open-source libraries such as OpenCV, Dlib, and NumPy to detect facial landmarks and compute measurements that classify the face shape into categories like Oval, Round, Square, Diamond, and more.

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

\```bash
pip install streamlit opencv-python dlib face-recognition imutils numpy
\```

Additionally, you will need the **Dlib** facial landmark model, which can be downloaded [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).

Place the downloaded file in your project directory.

## Installation

1. **Clone the repository**:
   \```bash
   git clone https://github.com/your-username/face-shape-detector.git
   cd face-shape-detector
   \```

2. **Install dependencies**:
   Make sure you have Python installed, and then install the required Python libraries using the following command:
   \```bash
   pip install -r requirements.txt
   \```

3. **Download the facial landmark model**:
   Download the [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2) and place it in the project root directory.

4. **Run the app**:
   Start the Streamlit app by running the following command:
   \```bash
   streamlit run app.py
   \```

## Usage

1. Check the box to enable the camera.
2. Capture an image by using the "Capture Image" button.
3. The app will detect facial landmarks and overlay them on your picture.
4. Based on the facial proportions, it will classify your face shape and provide an explanation of the decision.

## Face Shapes and Classification

The app detects the following face shapes:

- **Oval**: The face length is longer than the cheekbones, the forehead is wider than the jawline, and the jawline is rounded.
- **Round**: The face length and cheekbones are approximately equal, with a softer jawline.
- **Square**: The face length, cheekbones, and jawline are relatively equal, with a sharp and angular jawline.
- **Heart**: The forehead is wider than the cheekbones and jawline, with a narrow, pointed chin.
- **Diamond**: The face length is the largest, followed by cheekbones, forehead, and then jawline. The chin is pointed.
- **Oblong**: The face length is noticeably greater than the cheekbones, forehead, and jawline, with the face appearing long and narrow.
- **Triangular**: The jawline is wider than the cheekbones and forehead, with a pointed or square chin.
- **Rectangle**: Similar to square, but with a longer face length, giving it a rectangular appearance.

### Key Measurements for Classification:

- **Face Length**: Distance from the top of the forehead to the chin.
- **Cheekbone Width**: Distance between the outermost points of the cheekbones.
- **Jawline Width**: Distance between the outermost points of the jawline.
- **Forehead Width**: Distance across the forehead from temple to temple.

The face shape is determined by comparing these measurements and assessing the proportions between them.

## Future Enhancements

- Further refine the face shape classification for more accuracy.
- Add more face analysis features such as symmetry and proportion analysis.
- Allow users to download or share their face shape analysis.

## File Structure

\```
face-shape-detector/
│
├── app.py                     # Main Streamlit app
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── shape_predictor_68_face_landmarks.dat  # Dlib facial landmark model
\```

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for building a simple, fast framework for web applications.
- [Dlib](http://dlib.net/) for providing facial landmark detection.
- [OpenCV](https://opencv.org/) for powerful image processing.
- [Face Recognition](https://github.com/ageitgey/face_recognition) library for easy access to face landmark features.
