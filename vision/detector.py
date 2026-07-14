import cv2
# pyrefly: ignore [missing-import]
import dlib
import numpy as np
import os
import bz2
import urllib.request
from scipy.spatial import distance
from collections import deque

# Compute absolute paths to models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
CNN_MODEL_PATH = os.path.join(BASE_DIR, "models", "mmod_human_face_detector.dat")

class EngagementDetector:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(MODEL_PATH)
        self.cnn_detector = None # Load lazily if GPU mode is activated

        # Default Tunable thresholds
        self.EAR_THRESHOLD = 0.23
        self.HEAD_YAW_TOLERANCE = 40.0
        self.HEAD_PITCH_TOLERANCE = 18.0  # Head drop tolerance (degrees)
        self.YAWN_MAR_THRESHOLD = 0.52   # Mouth aspect ratio threshold for yawning
        
        self.SMOOTH_FRAMES = 7
        self.SLEEPY_EAR_THRESHOLD = 0.20
        self.SLEEPY_CONSEC_FRAMES = 15
        self.YAWN_CONSEC_FRAMES = 15       # Consecutive frames of wide mouth to flag yawning
        self.HEAD_DROP_CONSEC_FRAMES = 12   # Consecutive frames of head tilt to flag sleepy/drooping

        # Tracking state
        self.next_face_id = 0
        self.face_centroids = {}  # face_id -> (cx, cy)
        self.face_data = {}       # face_id -> dict with history

    def _ensure_cnn_model(self):
        if not os.path.exists(CNN_MODEL_PATH):
            print("Downloading CNN face detector model...")
            url = "http://dlib.net/files/mmod_human_face_detector.dat.bz2"
            try:
                os.makedirs(os.path.dirname(CNN_MODEL_PATH), exist_ok=True)
                req = urllib.request.Request(
                    url, 
                    headers={'User-Agent': 'Mozilla/5.0'}
                )
                with urllib.request.urlopen(req) as response:
                    data = response.read()
                    decompressed = bz2.decompress(data)
                    with open(CNN_MODEL_PATH, 'wb') as f:
                        f.write(decompressed)
                print("CNN face detector model downloaded successfully!")
            except Exception as e:
                print(f"Error downloading CNN model: {e}")

    def reset_calibration(self, face_id=None):
        """Clears calibration history to trigger recalibration."""
        if face_id is not None:
            if face_id in self.face_data:
                self.face_data[face_id]["calibration_yaw_samples"] = []
                self.face_data[face_id]["calibration_pitch_samples"] = []
                self.face_data[face_id]["baseline_yaw"] = 0.0
                self.face_data[face_id]["baseline_pitch"] = 0.0
                self.face_data[face_id]["is_calibrated"] = False
        else:
            for fid in self.face_data:
                self.face_data[fid]["calibration_yaw_samples"] = []
                self.face_data[fid]["calibration_pitch_samples"] = []
                self.face_data[fid]["baseline_yaw"] = 0.0
                self.face_data[fid]["baseline_pitch"] = 0.0
                self.face_data[fid]["is_calibrated"] = False

    def eye_aspect_ratio(self, eye):
        A = distance.euclidean(eye[1], eye[5])
        B = distance.euclidean(eye[2], eye[4])
        C = distance.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)

    def mouth_aspect_ratio(self, shape):
        # Inner lip is shape[60:68] (8 points)
        mouth = shape[60:68]
        A = distance.euclidean(mouth[1], mouth[7]) # 61 and 67
        B = distance.euclidean(mouth[2], mouth[6]) # 62 and 66
        C = distance.euclidean(mouth[3], mouth[5]) # 63 and 65
        D = distance.euclidean(mouth[0], mouth[4]) # 60 and 64
        if D == 0:
            return 0.0
        return (A + B + C) / (2.0 * D)

    def get_head_pose(self, shape, frame_size):
        # shape is (68, 2) numpy array
        image_points = np.array([
            shape[30],  # Nose tip
            shape[8],   # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corner
            shape[48],  # Left mouth corner
            shape[54]   # Right mouth corner
        ], dtype="double")

        model_points = np.array([
            (0.0, 0.0, 0.0),             # Nose tip
            (0.0, -330.0, -65.0),        # Chin
            (-225.0, 170.0, -135.0),     # Left eye left corner
            (225.0, 170.0, -135.0),      # Right eye right corner
            (-150.0, -150.0, -125.0),    # Left mouth corner
            (150.0, -150.0, -125.0)      # Right mouth corner
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
            return 0.0, 0.0

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        pose_matrix = cv2.hconcat((rotation_matrix, translation_vector))
        _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_matrix)
        yaw = float(euler_angles[1])   # left/right
        pitch = float(euler_angles[0]) # up/down
        return yaw, pitch

    def compute_engagement_score(self, ear, yaw, pitch, ear_threshold, yaw_tolerance, pitch_tolerance):
        eye_score = np.clip((ear / ear_threshold) * 50, 0, 50)
        
        # Combine yaw and pitch deviations for a composite head pose score (50 points total)
        yaw_score = max(0, 25 - (abs(yaw) / yaw_tolerance) * 25)
        pitch_score = max(0, 25 - (abs(pitch) / pitch_tolerance) * 25)
        return eye_score + yaw_score + pitch_score

    def analyze_frame(self, frame, ear_threshold=None, yaw_tolerance=None, 
                      sleepy_ear_threshold=None, sleepy_consec_frames=None,
                      draw_landmarks=True, use_cnn=False):
        # Set OpenCL to speed up OpenCV if supported
        cv2.ocl.setUseOpenCL(True)

        # Override dynamic thresholds if provided
        eth = ear_threshold if ear_threshold is not None else self.EAR_THRESHOLD
        ytol = yaw_tolerance if yaw_tolerance is not None else self.HEAD_YAW_TOLERANCE
        seth = sleepy_ear_threshold if sleepy_ear_threshold is not None else self.SLEEPY_EAR_THRESHOLD
        scf = sleepy_consec_frames if sleepy_consec_frames is not None else self.SLEEPY_CONSEC_FRAMES

        frame_size = frame.shape
        h, w = frame_size[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Scale down frame for face detection
        target_width = 320
        scale_factor = 1.0
        if w > target_width:
            scale_factor = target_width / w
            small_gray = cv2.resize(gray, (target_width, int(h * scale_factor)))
        else:
            small_gray = gray

        # 2. Run Face Detection
        if use_cnn:
            if self.cnn_detector is None:
                self._ensure_cnn_model()
                if os.path.exists(CNN_MODEL_PATH):
                    try:
                        self.cnn_detector = dlib.cnn_face_detection_model_v1(CNN_MODEL_PATH)
                    except Exception as e:
                        print(f"Failed to load CNN detector: {e}. Falling back to HOG CPU detector.")
            
            if self.cnn_detector is not None:
                detections = self.cnn_detector(small_gray)
                faces = [d.rect for d in detections]
            else:
                faces = self.detector(small_gray)
        else:
            faces = self.detector(small_gray)

        # 3. Centroid Tracking
        current_centroids = []
        for face in faces:
            left = int(face.left() / scale_factor)
            top = int(face.top() / scale_factor)
            right = int(face.right() / scale_factor)
            bottom = int(face.bottom() / scale_factor)
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            current_centroids.append((cx, cy))

        current_face_ids = []
        for cx, cy in current_centroids:
            matched_id = None
            min_dist = float('inf')
            for fid, (prev_cx, prev_cy) in self.face_centroids.items():
                d = np.hypot(cx - prev_cx, cy - prev_cy)
                if d < min_dist and d < 120:  # 120 pixel matching radius
                    min_dist = d
                    matched_id = fid
            
            if matched_id is not None:
                current_face_ids.append(matched_id)
                self.face_centroids[matched_id] = (cx, cy)
            else:
                new_id = self.next_face_id
                self.next_face_id += 1
                current_face_ids.append(new_id)
                self.face_centroids[new_id] = (cx, cy)
                self.face_data[new_id] = {
                    "ear_history": deque(maxlen=self.SMOOTH_FRAMES),
                    "yaw_history": deque(maxlen=self.SMOOTH_FRAMES),
                    "pitch_history": deque(maxlen=self.SMOOTH_FRAMES),
                    "sleepy_counter": 0,
                    "yawn_counter": 0,
                    "head_drop_counter": 0,
                    "last_seen_frames": 0,
                    # Calibration state per-face
                    "is_calibrated": False,
                    "baseline_yaw": 0.0,
                    "baseline_pitch": 0.0,
                    "calibration_yaw_samples": [],
                    "calibration_pitch_samples": []
                }

        # Cleanup disappeared faces
        for fid in list(self.face_centroids.keys()):
            if fid not in current_face_ids:
                self.face_data[fid]["last_seen_frames"] += 1
                if self.face_data[fid]["last_seen_frames"] > 30:  # 30 frames timeout
                    del self.face_centroids[fid]
                    del self.face_data[fid]
            else:
                self.face_data[fid]["last_seen_frames"] = 0

        sleepy_count = 0
        yawning_count = 0
        distracted_count = 0
        engaged_count = 0
        total_engagement_score = 0.0
        annotated_frame = frame.copy()

        # 4. Process landmarks and stats on full resolution
        for i, face in enumerate(faces):
            fid = current_face_ids[i]
            face_info = self.face_data[fid]
            
            # Scale coordinates back up to full-resolution for landmarks shape predictor
            left = int(face.left() / scale_factor)
            top = int(face.top() / scale_factor)
            right = int(face.right() / scale_factor)
            bottom = int(face.bottom() / scale_factor)
            scaled_face = dlib.rectangle(left, top, right, bottom)

            shape_obj = self.predictor(gray, scaled_face)
            shape = np.array([[p.x, p.y] for p in shape_obj.parts()])

            # Extract eyes and EAR
            left_eye = shape[42:48]
            right_eye = shape[36:42]
            leftEAR = self.eye_aspect_ratio(left_eye)
            rightEAR = self.eye_aspect_ratio(right_eye)
            ear = (leftEAR + rightEAR) / 2.0

            # Get Mouth Aspect Ratio (MAR)
            mar = self.mouth_aspect_ratio(shape)

            # Get Raw Yaw & Pitch
            raw_yaw, raw_pitch = self.get_head_pose(shape, frame_size)

            # ----------------------------------------------
            # Per-Face Calibration Routine
            # ----------------------------------------------
            if not face_info.get("is_calibrated", False):
                face_info.setdefault("calibration_yaw_samples", []).append(raw_yaw)
                face_info.setdefault("calibration_pitch_samples", []).append(raw_pitch)
                
                # Render "Calibrating..." frame
                status = "Calibrating..."
                color = (255, 255, 0) # Cyan
                engagement_score = 100.0
                
                samples_collected = len(face_info["calibration_yaw_samples"])
                if samples_collected >= 15:
                    face_info["baseline_yaw"] = np.mean(face_info["calibration_yaw_samples"])
                    face_info["baseline_pitch"] = np.mean(face_info["calibration_pitch_samples"])
                    face_info["is_calibrated"] = True
                    print(f"Face ID {fid} Calibrated! Baseline Yaw: {face_info['baseline_yaw']:.1f}°, Pitch: {face_info['baseline_pitch']:.1f}°")
                
                calibrated_yaw = 0.0
                calibrated_pitch = 0.0
            else:
                # Calculate deviations relative to baseline
                calibrated_yaw = raw_yaw - face_info["baseline_yaw"]
                calibrated_pitch = raw_pitch - face_info["baseline_pitch"]

            # Update smoothed histories
            face_info["ear_history"].append(ear)
            face_info["yaw_history"].append(calibrated_yaw)
            face_info["pitch_history"].append(calibrated_pitch)

            smoothed_ear = np.mean(face_info["ear_history"])
            smoothed_yaw = np.mean(face_info["yaw_history"])
            smoothed_pitch = np.mean(face_info["pitch_history"])

            # Process state checks ONLY if calibrated
            if face_info.get("is_calibrated", False):
                # ----------------------------------------------
                # Sleepy Eyes Detection
                # ----------------------------------------------
                if smoothed_ear < seth:
                    face_info["sleepy_counter"] += 1
                else:
                    face_info["sleepy_counter"] = max(0, face_info["sleepy_counter"] - 1)

                is_eye_sleepy = face_info["sleepy_counter"] >= scf

                # ----------------------------------------------
                # Head Drooping Detection
                # ----------------------------------------------
                if abs(smoothed_pitch) > self.HEAD_PITCH_TOLERANCE:
                    face_info["head_drop_counter"] += 1
                else:
                    face_info["head_drop_counter"] = max(0, face_info["head_drop_counter"] - 1)

                is_head_drop = face_info["head_drop_counter"] >= self.HEAD_DROP_CONSEC_FRAMES

                # ----------------------------------------------
                # Yawning Detection
                # ----------------------------------------------
                if mar > self.YAWN_MAR_THRESHOLD:
                    face_info["yawn_counter"] += 1
                else:
                    face_info["yawn_counter"] = max(0, face_info["yawn_counter"] - 1)

                is_yawning = face_info["yawn_counter"] >= self.YAWN_CONSEC_FRAMES

                # Classification
                if is_eye_sleepy or is_head_drop:
                    engagement_score = 0.0
                    status = "Sleepy"
                    if is_head_drop and not is_eye_sleepy:
                        status = "Head Drop"
                    color = (0, 0, 255) # Red
                    sleepy_count += 1
                elif is_yawning:
                    engagement_score = 20.0
                    status = "Yawning"
                    color = (255, 0, 255) # Magenta
                    yawning_count += 1
                else:
                    engagement_score = self.compute_engagement_score(smoothed_ear, smoothed_yaw, smoothed_pitch, eth, ytol, self.HEAD_PITCH_TOLERANCE)
                    if engagement_score > 70:
                        status = "Engaged"
                        color = (0, 255, 0) # Green
                        engaged_count += 1
                    else:
                        status = "Distracted"
                        color = (0, 165, 255) # Orange
                        distracted_count += 1

            total_engagement_score += engagement_score

            # Draw annotations
            x, y, w, h = scaled_face.left(), scaled_face.top(), scaled_face.width(), scaled_face.height()
            
            # Double line bounding box
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 0), 4)
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, 2)

            # Top label box: ID, status, and engagement score
            if not face_info.get("is_calibrated", False):
                label = f"ID:{fid} Calibrating ({15 - len(face_info['calibration_yaw_samples'])}f left)"
            else:
                label = f"ID:{fid} {status} ({int(engagement_score)})"
                
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = 0.55
            thickness = 2
            (label_w, label_h), baseline = cv2.getTextSize(label, font, scale, thickness)
            cv2.rectangle(annotated_frame, (x, y - label_h - 12), (x + label_w + 10, y), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (x, y - label_h - 12), (x + label_w + 10, y), color, 1)
            cv2.putText(annotated_frame, label, (x + 5, y - 6), font, scale, (255, 255, 255), thickness - 1, cv2.LINE_AA)

            # Bottom info box: Live metrics for calibration (EAR, MAR, Calibrated Yaw, Calibrated Pitch)
            metrics_text = f"EAR:{smoothed_ear:.2f} MAR:{mar:.2f} Y:{smoothed_yaw:.0f} P:{smoothed_pitch:.0f}"
            (m_w, m_h), m_base = cv2.getTextSize(metrics_text, font, 0.45, 1)
            cv2.rectangle(annotated_frame, (x, y + h), (x + m_w + 10, y + h + m_h + 8), (0, 0, 0), -1)
            cv2.rectangle(annotated_frame, (x, y + h), (x + m_w + 10, y + h + m_h + 8), (120, 120, 120), 1)
            cv2.putText(annotated_frame, metrics_text, (x + 5, y + h + m_h + 4), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            # Draw key landmarks (cyan dots)
            if draw_landmarks:
                for pt in shape:
                    cv2.circle(annotated_frame, (pt[0], pt[1]), 1, (255, 255, 0), -1)

        face_count = len(faces)
        class_engagement = 0
        if face_count > 0:
            class_engagement = int((total_engagement_score / (face_count * 100)) * 100)

        # Draw overall stats overlay in corner (semi-transparent background)
        if face_count > 0:
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (10, 10), (280, 50), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
            cv2.putText(annotated_frame, f"Class Engagement: {class_engagement}%", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        return {
            "frame": annotated_frame,
            "sleepy": sleepy_count,
            "yawning": yawning_count,
            "distracted": distracted_count,
            "engaged": engaged_count,
            "engagement": class_engagement,
            "num_faces": face_count
        }

if __name__ == "__main__":
    detector = EngagementDetector()
    cap = cv2.VideoCapture(0)
    print("Testing detector. Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        res = detector.analyze_frame(frame)
        cv2.imshow("Test Engagement Detector", res["frame"])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()