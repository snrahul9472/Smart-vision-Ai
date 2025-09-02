# import streamlit as st
# import cv2
# import numpy as np
# from datetime import datetime
# import time
# import tempfile
# import os
# from collections import defaultdict

# WEAPON_CLASSES = ["knife", "gun", "pistol", "rifle"]
# SUSPICIOUS_ITEMS = ["backpack", "bag"]

# # Constants
# CONFIDENCE_THRESHOLD = 0.5
# NMS_THRESHOLD = 0.4#Non max suppression is a technique 
# #used mainly in object detection that aims at selecting 
# # the best bounding box out of a set of overlapping boxes.
# LOITERING_TIME_THRESHOLD = 30
# AGGRESSIVE_MOVEMENT_THRESHOLD = 0.3
# ALERT_COOLDOWN = 30

# # Load YOLO model
# @st.cache_resource
# def load_yolo_model():
#     weights_path = "yolov4-tiny.weights"
#     config_path = "yolov4-tiny.cfg"
    
#     if not os.path.exists(weights_path):
#         st.error("YOLO weights file not found. Please download yolov4-tiny.weights.")
#         st.stop()
    
#     net = cv2.dnn.readNet(config_path, weights_path)
#     net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
#     net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
#     with open("coco.names", "r") as f:
#         classes = [line.strip() for line in f.readlines()]
    
#     layer_names = net.getLayerNames()
#     output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
#     return net, classes, output_layers

# # Detection and alert functions
# def detect_fights(frame, prev_frame, motion_history, fgbg):
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (21, 21), 0)
    
#     if prev_frame is None:
#         return False, 0, gray, motion_history
    
#     frame_delta = cv2.absdiff(prev_frame, gray)
#     thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
#     thresh = cv2.dilate(thresh, None, iterations=2)
    
#     contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
#     fight_detected = False
#     motion_intensity = 0
    
#     for contour in contours:
#         if cv2.contourArea(contour) < 500:
#             continue
            
#         (x, y, w, h) = cv2.boundingRect(contour)
#         motion_intensity += w * h
        
#         aspect_ratio = w / float(h)
#         if 0.5 < aspect_ratio < 2.0:
#             motion_history.append((x, y, w, h))
#             if len(motion_history) > 10:
#                 dx = np.std([m[0] for m in motion_history[-10:]])
#                 dy = np.std([m[1] for m in motion_history[-10:]])
#                 if dx > 15 and dy > 15:
#                     fight_detected = True
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    
#     motion_level = motion_intensity / (frame.shape[0] * frame.shape[1])
#     return fight_detected, motion_level, gray, motion_history

# def draw_detections(frame, detections, alerts):
#     for detection in detections:
#         label = detection['label']
#         confidence = detection['confidence']
#         box = detection['box']
        
#         color = (0, 255, 0)
#         if label in WEAPON_CLASSES:
#             color = (0, 0, 255)
#         elif label in SUSPICIOUS_ITEMS:
#             color = (0, 165, 255)
        
#         cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 2)
#         cv2.putText(frame, f"{label} {confidence:.2f}", (box[0], box[1]-5), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
#     for alert in alerts:
#         if alert['type'] == 'weapon_detected':
#             box = alert['box']
#             cv2.rectangle(frame, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 0, 255), 3)
#             cv2.putText(frame, f"WEAPON: {alert['label'].upper()}", 
#                        (box[0], box[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
#         if alert['type'] == 'fight_detected':
#             cv2.putText(frame, "FIGHT DETECTED!", (20, 50), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
#     return frame

# # Streamlit app
# def main():
#     st.title("Crime Prevention System")
#     st.sidebar.title("Settings")
    
#     input_source = st.sidebar.radio("Input Source", ["Webcam", "Upload Video"])
#     confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
#     nms_threshold = st.sidebar.slider("NMS Threshold", 0.1, 1.0, 0.4, 0.1)
    
#     net, classes, output_layers = load_yolo_model()
    
#     if input_source == "Webcam":
#         st.write("Using Webcam...")
#         cap = cv2.VideoCapture(1)
#     else:
#         uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
#         if uploaded_file is not None:
#             temp_file = tempfile.NamedTemporaryFile(delete=False)
#             temp_file.write(uploaded_file.read())
#             cap = cv2.VideoCapture(temp_file.name)
#         else:
#             st.warning("Please upload a video file.")
#             return
    
#     stframe = st.empty()
#     prev_frame = None
#     motion_history = []
#     fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
#     frame_count = 0
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         height, width = frame.shape[:2]
#         blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
#         net.setInput(blob)
#         outs = net.forward(output_layers)
        
#         class_ids = []
#         confidences = []
#         boxes = []
        
#         for out in outs:
#             for detection in out:
#                 scores = detection[5:]
#                 class_id = np.argmax(scores)
#                 confidence = scores[class_id]
                
#                 if confidence > confidence_threshold:
#                     box = detection[0:4] * np.array([width, height, width, height])
#                     (center_x, center_y, box_width, box_height) = box.astype("int")
#                     x = int(center_x - (box_width / 2))
#                     y = int(center_y - (box_height / 2))
                    
#                     boxes.append([x, y, int(box_width), int(box_height)])
#                     confidences.append(float(confidence))
#                     class_ids.append(class_id)
        
#         indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
#         detections = []
#         if len(indices) > 0:
#             for i in indices.flatten():
#                 detections.append({
#                     'class_id': class_ids[i],
#                     'label': classes[class_ids[i]],
#                     'confidence': confidences[i],
#                     'box': boxes[i]
#                 })
        
#         fight_detected, motion_level, prev_frame, motion_history = detect_fights(
#             frame, prev_frame, motion_history, fgbg
#         )
        
#         alerts = []
#         if fight_detected:
#             alerts.append({"type": "fight_detected", "level": motion_level})
        
#         frame = draw_detections(frame, detections, alerts)
#         stframe.image(frame, channels="BGR")
        
#         frame_count += 1
    
#     cap.release()

# if __name__ == "__main__":
#     main()






#updated

import streamlit as st
import cv2
import numpy as np
from datetime import datetime
import time
import tempfile
import os
from collections import defaultdict
import requests
from threading import Thread
import queue
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
WEAPON_CLASSES = ["knife", "gun", "pistol", "rifle"]
SUSPICIOUS_ITEMS = ["backpack", "bag"]
CONFIDENCE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.4
LOITERING_TIME_THRESHOLD = 30
AGGRESSIVE_MOVEMENT_THRESHOLD = 0.3
ALERT_COOLDOWN = 30

# Initialize session state for alerts
def initialize_session_state():
    if 'alert_status' not in st.session_state:
        st.session_state.alert_status = None
    if 'alert_error' not in st.session_state:
        st.session_state.alert_error = None
    if 'last_alert_time' not in st.session_state:
        st.session_state.last_alert_time = 0
    if 'alert_queues' not in st.session_state:
        st.session_state.alert_queues = []
    if 'use_env_credentials' not in st.session_state:
        st.session_state.use_env_credentials = True
    if 'telegram_bot_token' not in st.session_state:
        st.session_state.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', "")
    if 'telegram_chat_id' not in st.session_state:
        st.session_state.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', "")
    if 'enable_telegram' not in st.session_state:
        st.session_state.enable_telegram = False
    if 'processing_active' not in st.session_state:
        st.session_state.processing_active = False
    if 'video_capture' not in st.session_state:
        st.session_state.video_capture = None

# Call the initialization function
initialize_session_state()

# Load YOLO model
@st.cache_resource
def load_yolo_model():
    weights_path = "yolov4-tiny.weights"
    config_path = "yolov4-tiny.cfg"
    
    if not os.path.exists(weights_path):
        st.error("YOLO weights file not found. Please download yolov4-tiny.weights.")
        st.stop()
    
    net = cv2.dnn.readNet(config_path, weights_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    
    return net, classes, output_layers

# Telegram Alert Function
def send_telegram_alert(bot_token=None, chat_id=None, result_queue=None, alert_type="fight"):
    if not all([bot_token, chat_id, result_queue]):
        result_queue.put(('error', "Missing Telegram credentials"))
        return

    try:
        if alert_type == "fight":
            message = "🚨 Fight Alert triggered!"
        else:
            message = f"🚨 Weapon Alert triggered! ({alert_type})"
            
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        params = {
            'chat_id': chat_id,
            'text': message
        }

        response = requests.post(url, params=params)
        if response.status_code == 200:
            result_queue.put(('success', "Telegram alert sent successfully"))
        else:
            result_queue.put(('error', f"Telegram API error: {response.text}"))
    except Exception as e:
        result_queue.put(('error', f"Exception occurred: {str(e)}"))

# Detection and alert functions
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
        elif label in SUSPICIOUS_ITEMS:
            color = (0, 165, 255)
        
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

def process_video(cap, net, classes, output_layers, confidence_threshold, nms_threshold):
    stframe = st.empty()
    prev_frame = None
    motion_history = []
    fgbg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=False)
    
    while st.session_state.processing_active and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        class_ids = []
        confidences = []
        boxes = []
        
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                if confidence > confidence_threshold:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, box_width, box_height) = box.astype("int")
                    x = int(center_x - (box_width / 2))
                    y = int(center_y - (box_height / 2))
                    
                    boxes.append([x, y, int(box_width), int(box_height)])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
        
        detections = []
        if len(indices) > 0:
            for i in indices.flatten():
                detections.append({
                    'class_id': class_ids[i],
                    'label': classes[class_ids[i]],
                    'confidence': confidences[i],
                    'box': boxes[i]
                })
        
        fight_detected, motion_level, prev_frame, motion_history = detect_fights(
            frame, prev_frame, motion_history, fgbg
        )
        
        alerts = []
        if fight_detected:
            alerts.append({"type": "fight_detected", "level": motion_level})
            
            # Send Telegram alert if fight detected and cooldown has passed
            if (st.session_state.enable_telegram and 
                time.time() - st.session_state.last_alert_time > ALERT_COOLDOWN):
                
                bot_token = st.session_state.telegram_bot_token
                chat_id = st.session_state.telegram_chat_id
                
                if bot_token and chat_id:
                    alert_queue = queue.Queue()
                    Thread(
                        target=send_telegram_alert,
                        args=(bot_token, chat_id, alert_queue, "fight")
                    ).start()
                    
                    st.session_state.alert_queues.append(alert_queue)
                    st.session_state.last_alert_time = time.time()
                    st.toast("Fight detected! Telegram alert sent.")
                else:
                    st.warning("Telegram credentials missing")
        
        # Check for weapon detections
        weapon_detected = any(det['label'] in WEAPON_CLASSES for det in detections)
        if weapon_detected:
            weapon_detections = [det for det in detections if det['label'] in WEAPON_CLASSES]
            for weapon in weapon_detections:
                alerts.append({
                    "type": "weapon_detected",
                    "label": weapon['label'],
                    "confidence": weapon['confidence'],
                    "box": weapon['box']
                })
                
                # Send Telegram alert if weapon detected and cooldown has passed
                if (st.session_state.enable_telegram and 
                    time.time() - st.session_state.last_alert_time > ALERT_COOLDOWN):
                    
                    bot_token = st.session_state.telegram_bot_token
                    chat_id = st.session_state.telegram_chat_id
                    
                    if bot_token and chat_id:
                        alert_queue = queue.Queue()
                        Thread(
                            target=send_telegram_alert,
                            args=(bot_token, chat_id, alert_queue, weapon['label'])
                        ).start()
                        
                        st.session_state.alert_queues.append(alert_queue)
                        st.session_state.last_alert_time = time.time()
                        st.toast(f"Weapon ({weapon['label']}) detected! Telegram alert sent.")
                    else:
                        st.warning("Telegram credentials missing")
        
        # Check for alert results
        for q in list(st.session_state.alert_queues):
            try:
                status, message = q.get_nowait()
                if status == 'success':
                    st.session_state.alert_status = message
                else:
                    st.session_state.alert_error = message
                st.session_state.alert_queues.remove(q)
            except queue.Empty:
                pass
        
        if st.session_state.alert_status:
            st.success(st.session_state.alert_status)
            st.session_state.alert_status = None
        
        if st.session_state.alert_error:
            st.error(st.session_state.alert_error)
            st.session_state.alert_error = None
        
        frame = draw_detections(frame, detections, alerts)
        stframe.image(frame, channels="BGR", use_column_width=True)
    
    if st.session_state.video_capture:
        st.session_state.video_capture.release()
        st.session_state.video_capture = None

# Streamlit app
def main():
    st.set_page_config(
        page_title="Crime Prevention System",
        page_icon="🛡️",
        layout="wide"
    )
    
    st.title("🛡️ Crime Prevention System")
    st.markdown("""
    <style>
    .stAlert {padding: 10px; border-radius: 5px;}
    .st-b7 {color: white;}
    .css-1aumxhk {background-color: #0e1117;}
    </style>
    """, unsafe_allow_html=True)
    
    with st.sidebar:
        st.title("Settings")
        input_source = st.radio("Input Source", ["Webcam", "Upload Video"], index=1)
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 1.0, 0.5, 0.1)
        nms_threshold = st.slider("NMS Threshold", 0.1, 1.0, 0.4, 0.1)
        
        st.markdown("---")
        st.subheader("Alert Settings")
        st.session_state.enable_telegram = st.checkbox("Enable Telegram Alerts", value=False)
        
        if st.session_state.enable_telegram:
            st.session_state.use_env_credentials = st.radio(
                "Telegram Credentials Source",
                ["Use .env file", "Enter manually"],
                index=0 if st.session_state.use_env_credentials else 1
            )

            if st.session_state.use_env_credentials:
                st.info("Using credentials from .env file")
                if st.session_state.telegram_bot_token:
                    st.write(f"Bot Token: {'*' * len(st.session_state.telegram_bot_token)}")
                if st.session_state.telegram_chat_id:
                    st.write(f"Chat ID: {st.session_state.telegram_chat_id}")
            else:
                st.session_state.telegram_bot_token = st.text_input("Telegram Bot Token", 
                                                                  value=st.session_state.telegram_bot_token,
                                                                  type="password")
                st.session_state.telegram_chat_id = st.text_input("Telegram Chat ID",
                                                                value=st.session_state.telegram_chat_id)
    
    net, classes, output_layers = load_yolo_model()
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        video_placeholder = st.empty()
        
        if st.session_state.processing_active:
            if st.button("⏹️ Stop Processing"):
                st.session_state.processing_active = False
                if st.session_state.video_capture:
                    st.session_state.video_capture.release()
                    st.session_state.video_capture = None
                st.experimental_rerun()
        else:
            if input_source == "Webcam":
                if st.button("🎥 Start Webcam"):
                    st.session_state.video_capture = cv2.VideoCapture(0)
                    if st.session_state.video_capture.isOpened():
                        st.session_state.processing_active = True
                        st.experimental_rerun()
                    else:
                        st.error("Could not open webcam")
            else:
                uploaded_file = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
                if uploaded_file is not None and st.button("▶️ Start Processing"):
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    temp_file.write(uploaded_file.read())
                    st.session_state.video_capture = cv2.VideoCapture(temp_file.name)
                    st.session_state.processing_active = True
                    st.experimental_rerun()
    
    with col2:
        log_placeholder = st.empty()
        
        if st.session_state.alert_status:
            log_placeholder.success(st.session_state.alert_status)
        if st.session_state.alert_error:
            log_placeholder.error(st.session_state.alert_error)
    
    if st.session_state.processing_active and st.session_state.video_capture:
        process_video(
            st.session_state.video_capture,
            net,
            classes,
            output_layers,
            confidence_threshold,
            nms_threshold
        )

if __name__ == "__main__":
    main()