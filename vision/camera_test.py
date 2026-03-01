import cv2

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, frame = cap.read()
if not ret:
    print("Failed to grab frame")
else:
    print("Camera works! Frame shape:", frame.shape)

cap.release()
