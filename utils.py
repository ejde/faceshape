import cv2
import numpy as np
import matplotlib.pyplot as plt

# todo: encapsulate all this functionality in a class
def visualize_landmarks(image, landmarks):
    plt.figure(figsize=(8, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    for idx, point in enumerate(landmarks):
        plt.scatter(point[0], point[1], c='yellow', s=10)
        plt.text(point[0], point[1], str(idx), fontsize=8, color='red')
    plt.axis('off')
    plt.show()

def euclidean_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def is_approximately_equal(a, b, tolerance=0.15):
    return abs(a - b) <= tolerance * max(abs(a), abs(b))

def are_measurements_approximately_equal(*args):
    return all(is_approximately_equal(a, args[0]) for a in args[1:])

# Function to calculate the key facial landmarks used for both analysis and drawing
def calculate_face_landmarks(landmarks):
    jaw_points = landmarks[4:13] #4:13

    # Calculate the midpoint between jawline point and outer eye corner
    left_cheekbone = (
        (landmarks[4][0] + landmarks[36][0]) / 2,
        (landmarks[4][1] + landmarks[36][1]) / 2
    )

    right_cheekbone = (
        (landmarks[12][0] + landmarks[45][0]) / 2,
        (landmarks[12][1] + landmarks[45][1]) / 2
    )

    chin = landmarks[8]

    # Estimated forehead points (using vertical shift)
    vertical_shift = (landmarks[27][1] - landmarks[8][1]) * 0.25  # Adjust the multiplier as needed
    forehead_left = (landmarks[17][0], landmarks[17][1] + vertical_shift)
    forehead_right = (landmarks[26][0], landmarks[26][1] + vertical_shift)
    forehead_top = (landmarks[27][0], landmarks[27][1] + vertical_shift)

    return {
        "jaw_points": jaw_points,
        "left_cheekbone": left_cheekbone,
        "right_cheekbone": right_cheekbone,
        "chin": chin,
        "forehead_left": forehead_left,
        "forehead_right": forehead_right,
        "forehead_top": forehead_top
    }

# Function to analyze the face shape based on the calculated landmarks
# todo: clean up the logic, add more features to assess
def classify_shape(jaw_width, cheekbone_width, forehead_width, face_length):
    matches = []

    if forehead_width > cheekbone_width and is_approximately_equal(cheekbone_width, jaw_width):
        matches.append(("Heart", "Forehead is wider than cheekbones and jawline."))

    if is_approximately_equal(cheekbone_width, face_length) and cheekbone_width > jaw_width:
        matches.append(("Diamond", "Cheekbones are the widest part of the face."))

    if jaw_width > cheekbone_width and cheekbone_width > forehead_width:
        matches.append(("Triangle", "Jawline is wider than cheekbones and forehead."))

    if are_measurements_approximately_equal(face_length, cheekbone_width, jaw_width):
        matches.append(("Square", "Face length, cheekbones, and jawline are approximately equal."))

    if face_length > cheekbone_width and is_approximately_equal(forehead_width, jaw_width):
        matches.append(("Oblong", "Face is longer with similar widths across the face."))

    if face_length < cheekbone_width and is_approximately_equal(cheekbone_width, jaw_width):
        matches.append(("Round", "Face width is greater than face length, with similar cheekbone and jaw widths."))

    if is_approximately_equal(forehead_width, jaw_width) and face_length > 1.3 * cheekbone_width:
        matches.append(("Rectangle", "Face length is greater, with similar forehead and jawline widths."))

    if not matches:
        matches.append(("Unknown", "Unable to classify face shape with the given measurements."))

    return matches

# Function to draw lines connecting landmarks for facial features and measurements
def measure_face(image, landmarks):
    # Get the calculated landmarks
    face_landmarks = calculate_face_landmarks(landmarks)

    jaw_points = face_landmarks['jaw_points']
    left_cheekbone = face_landmarks['left_cheekbone']
    right_cheekbone = face_landmarks['right_cheekbone']
    chin = face_landmarks['chin']
    forehead_left = face_landmarks['forehead_left']
    forehead_right = face_landmarks['forehead_right']
    forehead_top = face_landmarks['forehead_top']

    # draw_facial_features(image, jaw_points, left_cheekbone, right_cheekbone, chin, forehead_left, forehead_right, forehead_top)
    draw_measurement_lines(image, jaw_points, left_cheekbone, right_cheekbone, chin, forehead_left, forehead_right, forehead_top)

    jaw_width = euclidean_distance(jaw_points[0], jaw_points[-1])
    cheekbone_width = euclidean_distance(left_cheekbone, right_cheekbone)
    forehead_width = euclidean_distance(forehead_left, forehead_right)
    face_length = euclidean_distance(forehead_top, chin)

    return jaw_width, cheekbone_width, forehead_width, face_length

def draw_facial_features(image, jaw_points, left_cheekbone, right_cheekbone, chin, forehead_left, forehead_right, forehead_top):
    for i in range(len(jaw_points)-1):
        cv2.line(image, tuple(map(int, jaw_points[i])), tuple(map(int, jaw_points[i + 1])), (0, 255, 0), 2)

    cv2.line(image, tuple(map(int, left_cheekbone)), tuple(map(int, right_cheekbone)), (255, 0, 0), 2)

    cv2.line(image, tuple(map(int, forehead_left)), tuple(map(int, forehead_right)), (0, 0, 255), 2)
    cv2.line(image, tuple(map(int, forehead_left)), tuple(map(int, forehead_top)), (0, 0, 255), 2)
    cv2.line(image, tuple(map(int, forehead_right)), tuple(map(int, forehead_top)), (0, 0, 255), 2)

def draw_measurement_lines(image, jaw_points, left_cheekbone, right_cheekbone, chin, forehead_left, forehead_right, forehead_top):
    thickness = 1
    cv2.line(image, tuple(map(int, jaw_points[0])), tuple(map(int, jaw_points[-1])), (0, 255, 255), thickness)  # Jawline, Yellow
    cv2.line(image, tuple(map(int, left_cheekbone)), tuple(map(int, right_cheekbone)), (255, 255, 0), thickness)  # Cheekbone, Cyan
    cv2.line(image, tuple(map(int, forehead_left)), tuple(map(int, forehead_right)), (255, 0, 255), thickness)  # Forehead, Magenta
    cv2.line(image, tuple(map(int, chin)), tuple(map(int, forehead_top)), (0, 128, 255), thickness)  # Face length, Orange

