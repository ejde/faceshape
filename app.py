import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils

# Load pre-trained facial landmark detector globally
@st.cache_resource
def load_model():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return detector, predictor

# Load model
detector, predictor = load_model()

# Helper function to calculate Euclidean distance
def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)

# Helper function to analyze face shape
def analyze_face_shape(landmarks):
    # Extract important points for measurements
    jaw_points = landmarks[0:17]  # Jawline points
    left_cheekbone = landmarks[1]
    right_cheekbone = landmarks[15]
    chin = landmarks[8]
    forehead_left = landmarks[17]
    forehead_right = landmarks[26]
    forehead_top = landmarks[27]

    # Face length (distance from forehead top to chin)
    face_length = euclidean_distance(forehead_top, chin)
    
    # Forehead width (distance between left and right forehead points)
    forehead_width = euclidean_distance(forehead_left, forehead_right)

    # Cheekbone width (distance between left and right cheekbones)
    cheekbone_width = euclidean_distance(left_cheekbone, right_cheekbone)

    # Jawline width (distance between jawline points)
    jaw_width = euclidean_distance(jaw_points[0], jaw_points[16])

    # Ratio calculations for classifying face shapes
    if face_length > cheekbone_width and forehead_width > jaw_width and jaw_width < cheekbone_width:
        return "Oval", "Face length is greater than cheekbones, and forehead is wider than the jawline."
    elif face_length > cheekbone_width and cheekbone_width > jaw_width:
        return "Oblong", "Face length is noticeably longer than cheekbones and jawline, creating a longer appearance."
    elif cheekbone_width > face_length and cheekbone_width > jaw_width:
        return "Diamond", "Cheekbones are the widest part of the face, with a narrow jawline and forehead."
    elif face_length == cheekbone_width == jaw_width:
        return "Square", "Face length, cheekbones, and jawline are roughly equal in width, with a sharp jawline."
    elif face_length < cheekbone_width and jaw_width > forehead_width:
        return "Triangular", "Jawline is wider than cheekbones and forehead, giving a triangular appearance."
    elif jaw_width == forehead_width and face_length > cheekbone_width:
        return "Rectangle", "Face length is greater than cheekbones and jawline, with similar forehead and jawline widths."
    elif cheekbone_width > face_length and cheekbone_width == jaw_width:
        return "Round", "Face length is almost equal to the width of the cheekbones and jawline, creating a softer, round appearance."
    elif forehead_width > cheekbone_width > jaw_width and chin[1] < forehead_top[1]:
        return "Heart", "Forehead is wider than cheekbones and jawline, with a pointed chin."
    else:
        return "Unknown", "Unable to classify face shape with the given measurements."

# Streamlit interface
st.title("Face Shape Detector")

# Webcam Capture
st.subheader("Take a picture to analyze your face shape:")
camera = st.camera_input("Capture Image")

if camera:
    # Convert the captured image to OpenCV format
    file_bytes = np.asarray(bytearray(camera.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Detect face landmarks
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) > 0:
        # Get landmarks
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Analyze face shape based on landmarks
            face_shape, reason = analyze_face_shape(shape)

            # Display results
            st.write(f"Detected Face Shape: {face_shape}")
            st.write(f"Reason: {reason}")

            # Draw the overlay for the landmarks and guides
            for (x, y) in shape:
                cv2.circle(img, (x, y), 1, (0, 255, 0), -1)

            # Display image with landmarks
            st.image(img, channels="BGR")
    else:
        st.write("No face detected. Please try again.")
