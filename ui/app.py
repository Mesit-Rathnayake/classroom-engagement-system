import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import time
from collections import deque
from vision.detector import EngagementDetector

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Classroom Engagement Monitor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Classroom Engagement Monitoring Dashboard")
st.markdown("Real-time visual monitoring, distraction analysis, and engagement trends.")
st.markdown("---")

# ------------------ SESSION STATE ------------------
if "run" not in st.session_state:
    st.session_state.run = False

if "detector" not in st.session_state:
    st.session_state.detector = EngagementDetector()

if "engagement_history" not in st.session_state:
    st.session_state.engagement_history = deque(maxlen=100)

detector = st.session_state.detector

# ------------------ SIDEBAR CONTROLS & THRESHOLDS ------------------
st.sidebar.header("⚙️ Dashboard Controls")

start_button = st.sidebar.button("▶ Start Monitoring", width="stretch")
stop_button = st.sidebar.button("⏹ Stop Monitoring", width="stretch")

if start_button:
    st.session_state.run = True
    st.session_state.engagement_history.clear()

if stop_button:
    st.session_state.run = False
    if "cap" in st.session_state and st.session_state.cap is not None:
        try:
            st.session_state.cap.release()
        except Exception:
            pass
        st.session_state.cap = None

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Detection Thresholds")

ear_threshold = st.sidebar.slider(
    "Eye Aspect Ratio (EAR) Threshold",
    min_value=0.15,
    max_value=0.30,
    value=0.23,
    step=0.01,
    help="Higher values make detector more sensitive to eye narrowness/blinks."
)

sleepy_ear_threshold = st.sidebar.slider(
    "Sleepy EAR Threshold",
    min_value=0.10,
    max_value=0.25,
    value=0.20,
    step=0.01,
    help="Threshold below which eye closure indicates sleepiness."
)

sleepy_consec_frames = st.sidebar.slider(
    "Sleepy Consecutive Frames",
    min_value=5,
    max_value=50,
    value=15,
    step=1,
    help="Number of consecutive frames below Sleepy EAR to flag a student as sleepy."
)

yaw_tolerance = st.sidebar.slider(
    "Head Yaw Tolerance (deg)",
    min_value=10.0,
    max_value=90.0,
    value=40.0,
    step=5.0,
    help="Maximum head angle to left/right before flagging student as distracted."
)

draw_landmarks = st.sidebar.checkbox("Show Facial Landmarks", value=True)

use_cnn = st.sidebar.checkbox(
    "Use GPU Face Detection (dlib CNN)",
    value=False,
    help="Leverages your GPU for face detection using dlib's CNN detector. Downloads a 700KB model file on first run. Requires CUDA-enabled dlib."
)

# Clear history button in sidebar
if st.sidebar.button("🧹 Clear History"):
    st.session_state.engagement_history.clear()

if st.sidebar.button("🎯 Recalibrate Base Posture"):
    detector.reset_calibration()

# ------------------ TOP METRICS BAR ------------------
metric_cols = st.columns(5)

with metric_cols[0]:
    engagement_metric = st.empty()

with metric_cols[1]:
    engaged_metric = st.empty()

with metric_cols[2]:
    distracted_metric = st.empty()

with metric_cols[3]:
    yawning_metric = st.empty()

with metric_cols[4]:
    sleepy_metric = st.empty()

st.markdown("---")

# ------------------ MAIN LAYOUT ------------------
col_vid, col_chart = st.columns([5, 3])

video_placeholder = col_vid.empty()

with col_chart:
    st.subheader("📈 Engagement History Trend")
    chart_placeholder = st.empty()
    
    st.subheader("💡 System Insights")
    insight_placeholder = st.empty()

# Initialize empty state visuals
engagement_metric.metric("Class Engagement", "0%")
engaged_metric.metric("Engaged Students", 0)
distracted_metric.metric("Distracted Students", 0)
yawning_metric.metric("Yawning Students", 0)
sleepy_metric.metric("Sleepy Students", 0)

video_placeholder.info("Click 'Start Monitoring' in the sidebar to initiate webcam feed.")
chart_placeholder.info("No session data. Start camera to populate historical engagement chart.")
insight_placeholder.info("Waiting for data to generate insights...")

# ------------------ CAMERA LOOP ------------------
if st.session_state.run:
    # Use cached VideoCapture to prevent re-opening delay on slider reruns
    if "cap" not in st.session_state or st.session_state.cap is None:
        st.session_state.cap = cv2.VideoCapture(0)
    cap = st.session_state.cap

    if not cap.isOpened():
        st.error("❌ Cannot access camera. It might be locked by another process.")
        st.session_state.run = False
    else:
        video_placeholder.empty()
        chart_placeholder.empty()
        insight_placeholder.empty()
        
        frame_idx = 0
        try:
            while st.session_state.run:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to grab frame")
                    break

                result = detector.analyze_frame(
                    frame,
                    ear_threshold=ear_threshold,
                    yaw_tolerance=yaw_tolerance,
                    sleepy_ear_threshold=sleepy_ear_threshold,
                    sleepy_consec_frames=sleepy_consec_frames,
                    draw_landmarks=draw_landmarks,
                    use_cnn=use_cnn
                )

                # Convert and show frame immediately (for maximum smoothness)
                frame_rgb = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", width="stretch")

                # Throttle UI metrics, history charts, and insights updates (every 10 frames)
                # This drastically reduces rendering overhead and extends chart history span
                frame_idx += 1
                if frame_idx % 10 == 0:
                    engagement_metric.metric("Class Engagement", f'{result["engagement"]}%')
                    engaged_metric.metric("Engaged Students", result["engaged"])
                    distracted_metric.metric("Distracted Students", result["distracted"])
                    yawning_metric.metric("Yawning Students", result["yawning"])
                    sleepy_metric.metric("Sleepy Students", result["sleepy"])

                    # Update history deque and line chart
                    st.session_state.engagement_history.append(result["engagement"])
                    chart_placeholder.line_chart(list(st.session_state.engagement_history))

                    # Dynamic System Insights
                    if result["num_faces"] == 0:
                        insight_placeholder.warning("⚠️ No faces detected. Ensure camera is clear and students are visible.")
                    elif result["sleepy"] > 0:
                        insight_placeholder.error(f"🚨 ALERT: {result['sleepy']} student(s) identified as sleepy/fatigued (head drop or closed eyes). Consider a brief break.")
                    elif result["yawning"] > 0:
                        insight_placeholder.warning(f"🥱 ALERT: {result['yawning']} student(s) yawning. Class may be losing focus.")
                    elif result["engagement"] < 50:
                        insight_placeholder.warning("📉 Engagement has dropped below 50%. The class may be distracted or disengaged.")
                    else:
                        insight_placeholder.success("✅ Class is performing well! Average engagement is within optimal range.")

                time.sleep(0.01)
        finally:
            # Only release camera on explicit stop, keeping it active during slider adjustments
            if not st.session_state.run:
                if "cap" in st.session_state and st.session_state.cap is not None:
                    try:
                        st.session_state.cap.release()
                    except Exception:
                        pass
                    st.session_state.cap = None