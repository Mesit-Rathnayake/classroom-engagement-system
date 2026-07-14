import cv2
import time
import argparse
import os
import sys

# Add directory containing main.py to path just in case
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from detector import EngagementDetector

def main():
    parser = argparse.ArgumentParser(description="Classroom Engagement System - Native CLI Viewer")
    parser.add_argument("--use-cnn", action="store_true", help="Use GPU CNN face detector (requires CUDA-enabled dlib)")
    parser.add_argument("--hide-landmarks", action="store_true", help="Hide facial landmark dots from display")
    parser.add_argument("--ear", type=float, default=None, help="Override eye aspect ratio threshold")
    parser.add_argument("--yaw", type=float, default=None, help="Override head yaw tolerance (deg)")
    args = parser.parse_args()

    detector = EngagementDetector()
    
    # Apply CLI threshold overrides if specified
    if args.ear is not None:
        detector.EAR_THRESHOLD = args.ear
    if args.yaw is not None:
        detector.HEAD_YAW_TOLERANCE = args.yaw

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return

    print("==================================================")
    print("Classroom Engagement System: Native CLI Running")
    print("==================================================")
    print("Press 'q' in the camera window to quit.")
    if args.use_cnn:
        print("GPU Face Detection (dlib CNN) is active.")
    else:
        print("CPU Face Detection (dlib HOG) is active.")
    print("--------------------------------------------------")

    prev_time = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Analyze using unified engine
        result = detector.analyze_frame(
            frame,
            draw_landmarks=not args.hide_landmarks,
            use_cnn=args.use_cnn
        )

        # Calculate FPS
        current_time = time.time()
        fps = 1.0 / (current_time - prev_time)
        prev_time = current_time

        # Draw FPS on screen
        cv2.putText(result["frame"], f"FPS: {fps:.1f}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        # Draw current mode indicator
        mode_text = "MODE: GPU CNN" if args.use_cnn else "MODE: CPU HOG"
        cv2.putText(result["frame"], mode_text, (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

        # Render OpenCV native window
        cv2.imshow("Classroom Engagement System", result["frame"])

        # Throttle terminal logs (print once per second / 30 frames)
        frame_idx += 1
        if frame_idx % 30 == 0:
            print(f"[Stats] Active Faces: {result['num_faces']} | Engaged: {result['engaged']} | Distracted: {result['distracted']} | Sleepy: {result['sleepy']} | Yawning: {result['yawning']} | Class: {result['engagement']}%")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\nShutting down camera feed...")
            break
        elif key == ord('c'):
            print("\nRecalibrating base posture...")
            detector.reset_calibration()

    cap.release()
    cv2.destroyAllWindows()
    print("Goodbye!")

if __name__ == "__main__":
    main()