import cv2
import numpy as np

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

        self.jaw_width = self.euclidean_distance(self.jaw_points[0], self.jaw_points[-1]) * 0.90
        self.cheekbone_width = self.euclidean_distance(self.left_cheekbone, self.right_cheekbone)
        self.forehead_width = self.euclidean_distance(self.forehead_left, self.forehead_right)
        self.face_length = self.euclidean_distance(self.forehead_top, self.chin)
        self.face_width = self.euclidean_distance(self.landmarks[0], self.landmarks[16])
        self.face_width_length_ratio = self.face_width / self.face_length
        self.jawline_angle = self.calculate_jawline_angle()

    def euclidean_distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

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

        def is_approximately_equal(a, b, tolerance=0.10):
            return abs(a - b) <= tolerance * max(abs(a), abs(b))

        def are_measurements_approximately_equal(*args):
            return all(self.is_approximately_equal(a, args[0]) for a in args[1:])
        
        # Oblong: Face length is significantly longer than other measurements, and forehead, cheekbones, jawline are similar
        if self.face_length >= 1.2 * max(self.forehead_width, self.cheekbone_width, self.jaw_width):
            if is_approximately_equal(self.forehead_width, self.cheekbone_width) or is_approximately_equal(self.cheekbone_width, self.jaw_width):
                matches.append(("Oblong", "Your face length is significantly longer than your other measurements, and your forehead, cheekbones, and jawline are almost equal in size."))

        # Square: All measurements are similar, with sharp jawline angles
        if is_approximately_equal(self.face_length, self.jaw_width) and is_approximately_equal(self.jaw_width, self.forehead_width) and is_approximately_equal(self.forehead_width, self.cheekbone_width) and self.jawline_angle < 116:
            matches.append(("Square", "Your face, jawline, forehead, and cheekbones are all similar in length. Meanwhile, your jawbone angles are sharp and not curved or round."))

        # Oval: Face is longer than cheekbones, forehead is wider than jawline, and jawline is rounded
        if self.face_length >= 1.2 * self.cheekbone_width and self.forehead_width >= 1.05 * self.jaw_width and self.jawline_angle >= 116:
            matches.append(("Oval", "Your face is longer than the width of your cheekbones, and your forehead is wider than your jawline. Meanwhile, your jawline angles are rounded."))

        # Round: Face length and cheekbones are similar, forehead and jawline are similar, jawline is rounded
        if is_approximately_equal(self.face_length, self.cheekbone_width) and is_approximately_equal(self.forehead_width, self.jaw_width) and self.jawline_angle >= 116:
            matches.append(("Round", "Your face length and cheekbones are similar, and your forehead and jawline measurements are similar. The jawline has soft, rounded angles."))

        # Diamond: Face length is longest, chin is pointy, measurements decrease from cheekbones to forehead to jawline
        if self.face_length >= self.cheekbone_width >= self.forehead_width >= self.jaw_width and self.jawline_angle < 116:
            matches.append(("Diamond", "Your face is the longest measurement, and your chin is pointy. The sizes decrease from cheekbones to forehead to jawline."))

        # Triangular: Jawline is wider than cheekbones, which are wider than forehead
        if self.jaw_width >= self.cheekbone_width >= self.forehead_width:
            matches.append(("Triangular", "Your jawline is wider than your cheekbones, which are wider than your forehead."))

        # Heart: Forehead is wider than cheekbones and jawline, chin is pointy
        if self.forehead_width > self.cheekbone_width and is_approximately_equal(self.cheekbone_width, self.jaw_width) and self.jawline_angle < 116:
            matches.append(("Heart", "Your forehead is wider than your cheekbones and jawline, and your chin is pointy."))

        # Handle unknown case
        if not matches:
            matches.append(("Unknown", "Unable to classify face shape with the given measurements."))

        return matches


    # Function to draw lines connecting landmarks for facial features and measurements
    def measure_face(self):
        # self.draw_facial_features(image)
        self.draw_measurement_lines(self.image)
        return self.jaw_width, self.cheekbone_width, self.forehead_width, self.face_length, self.face_width_length_ratio, self.jawline_angle

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