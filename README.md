# Face Shape and Hairstyle Recommendation App

This Streamlit app allows users to determine their face shape using a camera or uploaded image, and provides relevant measurements (e.g., jawline width, cheekbone width, forehead width, face length). Users can either upload an image or take a photo using their webcam. The app also provides a reset button to clear previous inputs and results for a fresh start.

## Features

- **Upload Image or Take Photo**: Users can upload a photo or take one directly with their webcam.
- **Face Shape Analysis**: The app detects facial landmarks using a 68-point face landmark model and calculates measurements such as:
  - Jawline width
  - Cheekbone width
  - Forehead width
  - Face length
- **Facial Landmarks Visualization**: Draws lines over facial landmarks on the image to illustrate the detected features.
- **Results Table**: Displays a table of facial measurements calculated from the image.
- **Reset Functionality**: The reset button clears the current image and measurements and resets the app for a new input.

## Getting Started

### Prerequisites

Before running the app, make sure you have Python installed along with the required libraries:

1. **Python 3.7+**
2. **Pip** (Python package manager)

### Installation

1. Clone the repository:
    \```bash
    git clone https://github.com/your-repo/face-shape-app.git
    cd face-shape-app
    \```

2. Install the dependencies:
    \```bash
    pip install -r requirements.txt
    \```

3. Run the Streamlit app:
    \```bash
    streamlit run app.py
    \```

### Required Libraries

The following Python packages are required and listed in `requirements.txt`:

- `streamlit`
- `opencv-python`
- `pillow`
- `face-alignment`
- `numpy`
- `pandas`

To install them manually, run:
\```bash
pip install streamlit opencv-python pillow face-alignment numpy pandas
\```

### How to Use

1. **Upload an image**:
   - Click on the "Choose an image..." button and select an image file (JPEG, JPG, PNG).
   
2. **Take a photo**:
   - If you prefer to use the webcam, click the "Take a photo" button to capture a photo directly in the app.

3. **View results**:
   - Once the image is processed, the app will display the following:
     - The original image with facial landmarks overlaid.
     - A table with the calculated measurements for jawline width, cheekbone width, forehead width, and face length.

4. **Reset**:
   - To reset the app, click the **Reset** button. This will clear the current input and results, allowing you to upload a new image or take another photo.

### Facial Landmarks Used

The app uses the 68-point face landmark model for analysis:
- Jawline (Points 0-16)
- Cheekbones (Approximated)
- Forehead (Approximated using vertical shift)
- Chin (Point 8)

### Example Use Case

- Upload an image of yourself or take a photo using your webcam.
- The app will detect your face, overlay key facial landmarks, and display the measurements of your facial features.
- Use the measurements for understanding your face shape and finding hairstyle recommendations that suit your face shape.

## Reset Functionality

The app provides a reset button that allows you to clear all previously uploaded images or taken photos and reset all results. This is helpful when switching between images or when you want to start over with a new input.

### Known Limitations

- Currently, the app is designed for single-face detection. It might not work correctly if multiple faces are present in the uploaded image or photo.
- Face detection depends on lighting conditions and image quality. Ensure your image or webcam photo is well-lit for the best results.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
