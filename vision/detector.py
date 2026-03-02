import cv2
import dlib
import numpy as np
from scipy.spatial import distance

class EngagementDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(
            "vision/models/shape_predictor_68_face_landmarks.dat"
        )

        # Tunable thresholds
        self.EAR_THRESHOLD = 0.22
        self.SLEEP_FRAMES = 20
        self.eye_counter = 0

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def analyze_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        sleepy_count = 0
        attentive_count = 0

        for face in faces:
            shape = self.predictor(gray, face)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = shape[42:48]
            right_eye = shape[36:42]

            leftEAR = self.eye_aspect_ratio(left_eye)
            rightEAR = self.eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < self.EAR_THRESHOLD:
                self.eye_counter += 1
                if self.eye_counter >= self.SLEEP_FRAMES:
                    sleepy_count += 1
            else:
                self.eye_counter = 0
                attentive_count += 1

        total = sleepy_count + attentive_count
        engagement = 0

        if total > 0:
            engagement = int((attentive_count / total) * 100)

        return {
            "frame": frame,
            "sleepy": sleepy_count,
            "attentive": attentive_count,
            "engagement": engagement
        }