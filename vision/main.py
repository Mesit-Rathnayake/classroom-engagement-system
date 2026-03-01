import cv2
import dlib
import numpy as np
import os
from scipy.spatial import distance as dist
from collections import deque

# ==========================================================
# CONFIGURATION
# ==========================================================

EAR_THRESHOLD = 0.23
HEAD_YAW_TOLERANCE = 40  # max angle considered "forward"
SMOOTH_FRAMES = 7
DISTRACTION_FRAMES = 15
MAX_SCORE = 100

SLEEPY_EAR_THRESHOLD = 0.20
SLEEPY_CONSEC_FRAMES = 15  # must be below threshold for this many frames

# ==========================================================
# PATH SETUP
# ==========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")

# ==========================================================
# HELPER FUNCTIONS
# ==========================================================

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def get_head_pose(shape, frame_size):
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose
        (shape.part(8).x, shape.part(8).y),    # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye
        (shape.part(45).x, shape.part(45).y),  # Right eye
        (shape.part(48).x, shape.part(48).y),  # Left mouth
        (shape.part(54).x, shape.part(54).y)   # Right mouth
    ], dtype="double")

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    focal_length = frame_size[1]
    center = (frame_size[1] / 2, frame_size[0] / 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
    yaw = float(euler_angles[1])  # left/right
    return yaw

def compute_engagement_score(ear, yaw):
    """
    Weighted engagement score:
    - 50 points for eye openness (0 if closed)
    - 50 points for head yaw (0 if fully turned away)
    """
    eye_score = np.clip((ear / EAR_THRESHOLD) * 50, 0, 50)
    yaw_score = max(0, 50 - (abs(yaw) / HEAD_YAW_TOLERANCE) * 50)
    return eye_score + yaw_score

# ==========================================================
# INITIALIZATION
# ==========================================================

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)
cap = cv2.VideoCapture(0)

face_data = {}  # store per-face smoothing and sleepy counters

# ==========================================================
# MAIN LOOP
# ==========================================================

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_size = frame.shape
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    total_engagement = 0
    face_count = len(faces)

    for i, face in enumerate(faces):
        shape = predictor(gray, face)
        face_id = i

        # Initialize history if new face
        if face_id not in face_data:
            face_data[face_id] = {
                "ear_history": deque(maxlen=SMOOTH_FRAMES),
                "yaw_history": deque(maxlen=SMOOTH_FRAMES),
                "sleepy_counter": 0
            }

        # ==================================================
        # EYE ASPECT RATIO
        # ==================================================
        left_eye = np.array([(shape.part(n).x, shape.part(n).y) for n in range(36, 42)])
        right_eye = np.array([(shape.part(n).x, shape.part(n).y) for n in range(42, 48)])
        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0
        face_data[face_id]["ear_history"].append(ear)
        smoothed_ear = np.mean(face_data[face_id]["ear_history"])

        # ==================================================
        # HEAD POSE
        # ==================================================
        yaw = get_head_pose(shape, frame_size)
        face_data[face_id]["yaw_history"].append(yaw)
        smoothed_yaw = np.mean(face_data[face_id]["yaw_history"])

        # ==================================================
        # SLEEPY DETECTION
        # ==================================================
        if smoothed_ear < SLEEPY_EAR_THRESHOLD:
            face_data[face_id]["sleepy_counter"] += 1
        else:
            face_data[face_id]["sleepy_counter"] = 0

        is_sleepy = face_data[face_id]["sleepy_counter"] >= SLEEPY_CONSEC_FRAMES

        # ==================================================
        # ENGAGEMENT SCORE
        # ==================================================
        if is_sleepy:
            engagement_score = 0
        else:
            engagement_score = compute_engagement_score(smoothed_ear, smoothed_yaw)

        total_engagement += engagement_score

        # ==================================================
        # STATUS DISPLAY
        # ==================================================
        if is_sleepy:
            status = "Sleepy"
            color = (0, 0, 255)
        elif engagement_score > 70:
            status = "Engaged"
            color = (0, 255, 0)
        elif engagement_score > 40:
            status = "Distracted"
            color = (0, 165, 255)
        else:
            status = "Distracted"
            color = (0, 165, 255)

        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{status} ({int(engagement_score)})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # ==================================================
    # CLASS ENGAGEMENT DISPLAY
    # ==================================================
    class_engagement_percent = (total_engagement / (face_count * MAX_SCORE) * 100) if face_count > 0 else 0
    cv2.putText(frame,
                f"Class Engagement: {class_engagement_percent:.1f}%",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2)

    cv2.imshow("Classroom Engagement System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()