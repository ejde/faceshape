import cv2
import numpy as np
import matplotlib.pyplot as plt

class Face():

    def __init__(self, image, landmarks):
        self.image = image
        self.landmarks = landmarks
        self.face_landmarks = self.calculate_face_landmarks()

        self.jaw_points = self.face_landmarks['jaw_points']
        self.left_cheekbone = self.face_landmarks['left_cheekbone']
        self.right_cheekbone = self.face_landmarks['right_cheekbone']
        self.chin = self.face_landmarks['chin']
        self.forehead_left = self.face_landmarks['forehead_left']
        self.forehead_right = self.face_landmarks['forehead_right']
        self.forehead_top = self.face_landmarks['forehead_top']

        self.jaw_width = self.euclidean_distance(self.jaw_points[0], self.jaw_points[-1])
        self.cheekbone_width = self.euclidean_distance(self.left_cheekbone, self.right_cheekbone)
        self.forehead_width = self.euclidean_distance(self.forehead_left, self.forehead_right)
        self.face_length = self.euclidean_distance(self.forehead_top, self.chin)
        self.face_width = self.euclidean_distance(self.landmarks[0], self.landmarks[16])
        self.face_width_length_ratio = self.face_width / self.face_length
        self.jawline_angle = self.calculate_jawline_angle()

    def visualize_landmarks(self, image, landmarks):
        plt.figure(figsize=(8, 8))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        for idx, point in enumerate(landmarks):
            plt.scatter(point[0], point[1], c='yellow', s=10)
            plt.text(point[0], point[1], str(idx), fontsize=8, color='red')
        plt.axis('off')
        plt.show()

    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def is_approximately_equal(self, a, b, tolerance=0.10):
        return abs(a - b) <= tolerance * max(abs(a), abs(b))

    def are_measurements_approximately_equal(self, *args):
        return all(self.is_approximately_equal(a, args[0]) for a in args[1:])

    def calculate_jawline_angle(self):
        # Calculate vectors along the left and right jawlines
        left_vector = np.array(self.landmarks[8]) - np.array(self.landmarks[0])
        right_vector = np.array(self.landmarks[16]) - np.array(self.landmarks[8])
        
        # Normalize vectors
        left_vector = left_vector / np.linalg.norm(left_vector)
        right_vector = right_vector / np.linalg.norm(right_vector)
        
        # Calculate angle between vectors
        angle_rad = np.arccos(np.clip(np.dot(left_vector, right_vector), -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        return angle_deg
   
    
    # Function to calculate the key facial landmarks used for both analysis and drawing
    def calculate_face_landmarks(self):
        landmarks = self.landmarks

        jaw_points = self.landmarks[4:13] #4:13
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
        vertical_shift = (landmarks[8][1] - landmarks[27][1] ) * 0.25  # Adjust the multiplier as needed
        forehead_left = (landmarks[17][0], landmarks[17][1] - vertical_shift)
        forehead_right = (landmarks[26][0], landmarks[26][1] - vertical_shift)
        forehead_top = (
            landmarks[27][0],
            (forehead_left[1] + forehead_right[1]) / 2
        )
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
    def classify_face(self):
        matches = []

        if self.forehead_width > self.cheekbone_width and self.is_approximately_equal(self.cheekbone_width, self.jaw_width) and self.face_width_length_ratio < 0.8:
            matches.append(("Heart", "Forehead is wider than cheekbones and jawline, with a longer face."))
    
        if self.is_approximately_equal(self.cheekbone_width, self.face_length) and self.cheekbone_width > self.jaw_width:
            matches.append(("Diamond", "Cheekbones are the widest part of the face."))

        if self.jaw_width > self.cheekbone_width and self.cheekbone_width > self.forehead_width:
            matches.append(("Triangle", "Jawline is wider than cheekbones and forehead."))

        if self.face_length > self.cheekbone_width and self.is_approximately_equal(self.forehead_width, self.jaw_width):
            matches.append(("Oblong", "Face is longer with similar widths across the face."))
        
        if self.forehead_width > self.jaw_width and self.face_width_length_ratio > 0.9:
            matches.append(("Oval", "Face is longer with decreasing widths across the face."))

        if self.are_measurements_approximately_equal(self.jaw_width, self.cheekbone_width, self.forehead_width) and self.face_width_length_ratio > 0.9:
            if self.jawline_angle > 125:
                matches.append(("Round", "Face is almost as wide as it is long, with a rounded jawline."))
            else:
                matches.append(("Square", "Face is almost as wide as it is long, with a strong jawline."))
    
        if not matches:
            matches.append(("Unknown", "Unable to classify face shape with the given measurements."))

        return matches

    # Function to draw lines connecting landmarks for facial features and measurements
    def measure_face(self):
        # self.draw_facial_features(image)
        self.draw_measurement_lines(self.image)
        return self.jaw_width, self.cheekbone_width, self.forehead_width, self.face_length, self.face_width_length_ratio

    def draw_facial_features(self, image):
        for i in range(len(self.jaw_points)-1):
            cv2.line(image, tuple(map(int, self.jaw_points[i])), tuple(map(int, self.jaw_points[i + 1])), (0, 255, 0), 2)

        cv2.line(image, tuple(map(int, self.left_cheekbone)), tuple(map(int, self.right_cheekbone)), (255, 0, 0), 2)

        cv2.line(image, tuple(map(int, self.forehead_left)), tuple(map(int, self.forehead_right)), (0, 0, 255), 2)
        cv2.line(image, tuple(map(int, self.forehead_left)), tuple(map(int, self.forehead_top)), (0, 0, 255), 2)
        cv2.line(image, tuple(map(int, self.forehead_right)), tuple(map(int, self.forehead_top)), (0, 0, 255), 2)

    def draw_measurement_lines(self, image):
        thickness = 1
        cv2.line(image, tuple(map(int, self.jaw_points[0])), tuple(map(int, self.jaw_points[-1])), (0, 255, 255), thickness)  # Jawline, Yellow
        cv2.line(image, tuple(map(int, self.left_cheekbone)), tuple(map(int, self.right_cheekbone)), (255, 255, 0), thickness)  # Cheekbone, Cyan
        cv2.line(image, tuple(map(int, self.forehead_left)), tuple(map(int, self.forehead_right)), (255, 0, 255), thickness)  # Forehead, Magenta
        cv2.line(image, tuple(map(int, self.chin)), tuple(map(int, self.forehead_top)), (0, 128, 255), thickness)  # Face length, Orange

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        classification = self.classify_face()
        classification_str = ", ".join([f"{name}: {description}" for name, description in classification])
        return (
            f"Face Measurements:\n"
            f"  Jaw Width: {self.jaw_width:.2f}\n"
            f"  Cheekbone Width: {self.cheekbone_width:.2f}\n"
            f"  Forehead Width: {self.forehead_width:.2f}\n"
            f"  Face Length: {self.face_length:.2f}\n"
            f"  Face Width: {self.face_width:.2f}\n"
            f"  Width-Length Ratio: {self.face_width_length_ratio:.2f}\n"
            f"  Jawline Angle: {self.jawline_angle:.2f}\n"
            f"Classification:\n"
            f"  {classification_str}"
        )