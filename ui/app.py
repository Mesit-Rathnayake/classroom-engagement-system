import sys
import os

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import cv2
import time
from vision.detector import EngagementDetector

# ------------------ PAGE CONFIG ------------------
st.set_page_config(layout="wide")
st.title("🎓 Classroom Engagement Monitoring System")

# ------------------ SESSION STATE ------------------
if "run" not in st.session_state:
    st.session_state.run = False

# ------------------ SIDEBAR ------------------
st.sidebar.header("⚙️ Controls")

start_button = st.sidebar.button("▶ Start Camera")
stop_button = st.sidebar.button("⏹ Stop Camera")

if start_button:
    st.session_state.run = True

if stop_button:
    st.session_state.run = False

# ------------------ LAYOUT ------------------
col1, col2 = st.columns([3, 1])

video_placeholder = col1.empty()

with col2:
    st.subheader("📊 Live Stats")
    engagement_metric = st.empty()
    sleepy_metric = st.empty()
    attentive_metric = st.empty()

# ------------------ DETECTOR INIT ------------------
detector = EngagementDetector()

# ------------------ CAMERA LOOP ------------------
if st.session_state.run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Cannot access camera")
    else:
        while st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to grab frame")
                break

            result = detector.analyze_frame(frame)

            # Convert frame for Streamlit
            frame_rgb = cv2.cvtColor(result["frame"], cv2.COLOR_BGR2RGB)
            video_placeholder.image(frame_rgb, channels="RGB")

            # Update Metrics
            engagement_metric.metric("Engagement", f'{result["engagement"]}%')
            sleepy_metric.metric("Sleepy Students", result["sleepy"])
            attentive_metric.metric("Attentive Students", result["attentive"])

            time.sleep(0.03)

        cap.release()