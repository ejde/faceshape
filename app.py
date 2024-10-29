import streamlit as st
import cv2
import numpy as np
import dlib
from imutils import face_utils
from PIL import Image
import pandas as pd

# Load the hairstyle recommendation image
@st.cache_resource
def load_hairstyle_image():
    return Image.open("faceshape-hairstyles.jpg")

hairstyle_image = load_hairstyle_image()

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

# Function to draw lines connecting landmarks for facial features
def draw_facial_lines(image, landmarks):
    # Draw lines for jaw (points 0-16)
    for i in range(16):
        cv2.line(image, tuple(landmarks[i]), tuple(landmarks[i + 1]), (0, 255, 0), 2)

    # Draw lines for left cheekbone (points 1-5) and right cheekbone (points 11-15)
    for i in range(1, 5):
        cv2.line(image, tuple(landmarks[i]), tuple(landmarks[i + 1]), (255, 0, 0), 2)  # Left cheekbone
    for i in range(11, 15):
        cv2.line(image, tuple(landmarks[i]), tuple(landmarks[i + 1]), (255, 0, 0), 2)  # Right cheekbone

    # Draw lines for forehead (points 17-26)
    for i in range(17, 26):
        cv2.line(image, tuple(landmarks[i]), tuple(landmarks[i + 1]), (0, 0, 255), 2)  # Forehead

    return draw_measurement_lines(image, landmarks)

# Function to calculate and draw jawline width, cheekbone width, forehead width, and face length
def draw_measurement_lines(image, landmarks):
    # Thinner line width
    thickness = 1

    # Calculate distances for measurements
    jawline_width = np.linalg.norm(np.array(landmarks[0]) - np.array(landmarks[16]))
    cheekbone_width = np.linalg.norm(np.array(landmarks[1]) - np.array(landmarks[15]))
    forehead_width = np.linalg.norm(np.array(landmarks[17]) - np.array(landmarks[26]))
    chin = landmarks[8]
    forehead_center = landmarks[27]  # Assuming index 27 represents the middle of the forehead
    face_length = np.linalg.norm(np.array(chin) - np.array(forehead_center))

    # Draw the lines for measurements
    cv2.line(image, tuple(landmarks[0]), tuple(landmarks[16]), (0, 255, 255), thickness)  # Jawline
    cv2.line(image, tuple(landmarks[1]), tuple(landmarks[15]), (255, 255, 0), thickness)  # Cheekbone
    cv2.line(image, tuple(landmarks[17]), tuple(landmarks[26]), (255, 0, 255), thickness)  # Forehead
    cv2.line(image, tuple(chin), tuple(forehead_center), (0, 128, 255), thickness)  # Face length

    # Return the calculated measurements
    return jawline_width, cheekbone_width, forehead_width, face_length

# Function to display relevant hairstyle portion based on face shape
def show_recommended_styles(face_shape):
    st.write(f"Recommended Hairstyles for {face_shape} Face Shape")
    if face_shape == "Oval":
        st.image(hairstyle_image.crop((0, 0, 200, 600)))  # Example cropping coordinates
    elif face_shape == "Square":
        st.image(hairstyle_image.crop((200, 0, 400, 600)))
    elif face_shape == "Oblong":
        st.image(hairstyle_image.crop((400, 0, 600, 600)))
    elif face_shape == "Triangular":
        st.image(hairstyle_image.crop((600, 0, 800, 600)))
    elif face_shape == "Round":
        st.image(hairstyle_image.crop((800, 0, 1000, 600)))
    elif face_shape == "Diamond":
        st.image(hairstyle_image.crop((1000, 0, 1200, 600)))
    elif face_shape == "Heart":
        st.image(hairstyle_image.crop((1200, 0, 1400, 600)))
    else:
        st.write("No hairstyle recommendation available for this face shape.")

st.title("Face Shape Detector")

# Webcam Capture
st.subheader("Take a picture to analyze your face shape:")
camera = st.camera_input("Capture Image")

if camera:
    #test image
    #import io
    #test_image = Image.open("test.jpg")
    #byte_stream = io.BytesIO()
    #test_image.save(byte_stream, format='JPEG')
    #img_byte_array = bytearray(byte_stream.getvalue())

    img_byte_array = camera.read()

    # Convert the captured image to OpenCV format
    file_bytes = np.asarray(bytearray(img_byte_array), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    # Detect face landmarks
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    if len(rects) > 0:
        # Get landmarks
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Display image with landmarks
            jawline_width, cheekbone_width, forehead_width, face_length = draw_facial_lines(img, shape)
            st.image(img, channels="BGR", caption="Detected facial features with lines")

            data = {
                "Feature": ["Jawline Width", "Cheekbone Width", "Forehead Width", "Face Length"],
                "Measurement (pixels)": [
                    f"{jawline_width:.2f}", 
                    f"{cheekbone_width:.2f}", 
                    f"{forehead_width:.2f}", 
                    f"{face_length:.2f}"
                ]
            }
            df = pd.DataFrame(data)
            st.table(df)
            
            # Analyze face shape based on landmarks
            face_shape, reason = analyze_face_shape(shape)

            # Display results
            st.write(f"Detected Face Shape: {face_shape}")
            st.write(f"Reason: {reason}")
            
            # show_recommended_styles(face_shape)
    else:
        st.write("No face detected. Please try again.")
