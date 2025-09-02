
import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
import tempfile
import os
from collections import defaultdict
from ultralytics import YOLO

# Custom-trained weapon classes
WEAPON_CLASSES = [
    'Automatic_Rifle',
    'Bazooka',
    'Handgun',
    'Knife',
    'Grenade_Launcher',
    'Shotgun',
    'SMG',
    'Sniper',
    'Sword'
]

@st.cache_resource
def load_custom_model():
    model_path = "best.pt"
    if not os.path.exists(model_path):
        st.error("Custom YOLO model not found at 'best.pt'.")
        st.stop()
    return YOLO(model_path)

def detect_fights(frame, prev_frame, motion_history, fgbg):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
    if prev_frame is None:
        return False, 0, gray, motion_history
    
    frame_delta = cv2.absdiff(prev_frame, gray)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    fight_detected = False
    motion_intensity = 0
    
    for contour in contours:
        if cv2.contourArea(contour) < 500:
            continue
            
        (x, y, w, h) = cv2.boundingRect(contour)
        motion_intensity += w * h
        
        aspect_ratio = w / float(h)
        if 0.5 < aspect_ratio < 2.0:
            motion_history.append((x, y, w, h))
            if len(motion_history) > 10:
                dx = np.std([m[0] for m in motion_history[-10:]])
                dy = np.std([m[1] for m in motion_history[-10:]])
                if dx > 15 and dy > 15:
                    fight_detected = True
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
    motion_level = motion_intensity / (frame.shape[0] * frame.shape[1])
    return fight_detected, motion_level, gray, motion_history

def draw_detections(frame, detections, alerts):
    for detection in detections:
        label = detection['label']
        confidence = detection['confidence']
        box = detection['box']
        
        color = (0, 255, 0)
        if label in WEAPON_CLASSES:
            color = (0, 0, 255)
        
        cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
        cv2.putText(frame, f"{label} {confidence:.2f}", (box[0], box[1]-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    for alert in alerts:
        if alert['type'] == 'weapon_detected':
            box = alert['box']
            cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 3)
            cv2.putText(frame, f"WEAPON: {alert['label'].upper()}", 
                       (box[0], box[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if alert['type'] == 'fight_detected':
            cv2.putText(frame, "FIGHT DETECTED!", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    return frame

def main():
    st.title("Weapon & Fight Detection System")
    st.sidebar.title("Settings")
    
    input_source = st.sidebar.radio("Input Source", ["Webcam", "Upload Video"])
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
    
    model = load_custom_model()
    
    if input_source == "Webcam":
        st.write("Using Webcam...")
        cap = cv2.VideoCapture(1)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False)
            temp_file.write(uploaded_file.read())
            cap = cv2.VideoCapture(temp_file.name)
        else:
            st.warning("Please upload a video file.")
            return
    
    stframe = st.empty()
    prev_frame = None
    motion_history = []
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        detections = []

        for r in results.boxes:
            box = r.xyxy[0].cpu().numpy().astype(int)
            conf = float(r.conf[0])
            cls_id = int(r.cls[0])
            label = model.names[cls_id]

            if conf >= confidence_threshold:
                x, y, x2, y2 = box
                w, h = x2 - x, y2 - y

                detections.append({
                    'class_id': cls_id,
                    'label': label,
                    'confidence': conf,
                    'box': [x, y, w, h]
                })

        fight_detected, motion_level, prev_frame, motion_history = detect_fights(
            frame, prev_frame, motion_history, fgbg
        )

        alerts = []
        for det in detections:
            if det['label'] in WEAPON_CLASSES:
                alerts.append({'type': 'weapon_detected', 'label': det['label'], 'box': det['box']})

        if fight_detected:
            alerts.append({"type": "fight_detected", "level": motion_level})

        frame = draw_detections(frame, detections, alerts)
        stframe.image(frame, channels="BGR")
    
    cap.release()

if __name__ == "__main__":
    main()
