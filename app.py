import streamlit as st
import numpy as np
import cv2
import dlib
from imutils import face_utils
from PIL import Image
import pandas as pd
from face import Face

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
detector, predictor = load_model()

def process_image(image_data):
    img = np.array(image_data.convert('RGB'))  # Convert to RGB array
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # Convert to BGR for OpenCV
    
    # Detect face landmarks
    rects = detector(img, 0)

    if len(rects) > 0:
        for rect in rects:
            shape = predictor(img, rect)
            shape = face_utils.shape_to_np(shape)

            # Display image with landmarks
            f = Face(img, shape)
            jawline_width, cheekbone_width, forehead_width, face_length, face_width_length_ratio, jawline_angle = f.measure_face()
            st.markdown('#### Measurements')
            st.image(img, channels="BGR", caption="Detected facial features with lines")
            data = {
                "Feature": ["Jawline Width", "Cheekbone Width", "Forehead Width", "Face Length", "Width-Length Ratio", "Jawline Angle"],
                "Measurement (pixels)": [
                    f"{jawline_width:.2f}", 
                    f"{cheekbone_width:.2f}", 
                    f"{forehead_width:.2f}", 
                    f"{face_length:.2f}",
                    f"{face_width_length_ratio:.2f}",
                    f"{jawline_angle:.2f}"
                ]
            }

            df = pd.DataFrame(data)
            st.table(df)
            
            matches = f.classify_face()
            st.markdown('#### Possible Face Shapes')
            for shape, reason in matches:
                st.write(f"Detected Face Shape: {shape}, Reason: {reason}")
                # show_recommended_styles(face_shape)
    else:
        st.write("No face detected. Please try again.")

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

# *** MAIN ***
st.title("Face Shape Detector")

# Initialize session state to keep track of the input type (file or camera)
st.session_state.input_type = None
st.session_state.image_processed = False

st.subheader("Upload an image or take a photo")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Take a photo")

# Handle file upload
if uploaded_file is not None:
    if st.session_state.input_type != "file":
        # Reset state when switching to file upload
        st.session_state.input_type = "file"
        st.session_state.image_processed = False

    if not st.session_state.image_processed:
        uploaded_image = Image.open(uploaded_file)
        process_image(uploaded_image)
        st.session_state.image_processed = True

# Handle camera input
elif camera_image is not None:
    if st.session_state.input_type != "camera":
        # Reset state when switching to camera input
        st.session_state.input_type = "camera"
        st.session_state.image_processed = False

    if not st.session_state.image_processed:
        camera_image_pil = Image.open(camera_image)
        process_image(camera_image_pil)
        st.session_state.image_processed = True

# If no image is provided, display instructions
else:
    st.write("Please upload an image or take a photo using the camera.")

if st.button("Reset"):
    # Clear the session state and force a rerun of the app to reset everything
    for key in st.session_state.keys():
        del st.session_state[key]
    st.experimental_set_query_params()