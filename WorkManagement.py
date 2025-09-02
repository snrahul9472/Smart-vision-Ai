import cv2
import torch
import os
import pandas as pd
from collections import defaultdict
from datetime import datetime

# Load YOLOv5 model
def initialize_yolov5(model_path='yolov5s.pt'):
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    model.conf = 0.4  # Confidence threshold
    return model

# Process a single frame
def process_frame_yolov5(model, frame):
    results = model(frame)
    detections = results.pandas().xyxy[0]  # Get results as pandas DataFrame
    return detections

# Main workplace monitoring function
def monitor_workplace(model, cctv_source):
    attendance_log = defaultdict(list)
    ppe_violations = []
    movement_patterns = defaultdict(list)
    
    cap = cv2.VideoCapture(cctv_source)
    frame_count = 0
    skip_frames = 5
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        
        current_time = datetime.now()
        detections = process_frame_yolov5(model, frame)
        
        people_in_frame = []
        detected_ppe = {'helmet': False, 'vest': False, 'gloves': False}
        
        for _, row in detections.iterrows():
            label = row['name']
            confidence = row['confidence']
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            
            color = (0, 255, 0)  # Green
            if label == 'person':
                people_in_frame.append((x1, y1, x2 - x1, y2 - y1))
                color = (0, 0, 255)  # Red
            elif label in detected_ppe:
                detected_ppe[label] = True
            
            # Draw boxes and labels
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Evaluate PPE for each person
        for person in people_in_frame:
            person_id = hash(person)
            
            movement_patterns[person_id].append({
                'timestamp': current_time,
                'position': (person[0], person[1])
            })
            
            if not all(detected_ppe.values()):
                violation = {
                    'timestamp': current_time,
                    'person_id': person_id,
                    'missing_gear': [k for k, v in detected_ppe.items() if not v]
                }
                ppe_violations.append(violation)
                cv2.putText(frame, "PPE VIOLATION!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        cv2.imshow('Work Monitoring (YOLOv5)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    generate_reports(attendance_log, ppe_violations, movement_patterns)

# Generate reports
def generate_reports(attendance_log, ppe_violations, movement_patterns):
    attendance_df = pd.DataFrame([
        {'employee_id': emp_id, 'time_in': min(times), 'time_out': max(times)}
        for emp_id, times in attendance_log.items()
    ])
    
    ppe_df = pd.DataFrame(ppe_violations)
    
    movement_data = []
    for emp_id, positions in movement_patterns.items():
        for pos in positions:
            movement_data.append({
                'employee_id': emp_id,
                'timestamp': pos['timestamp'],
                'x_position': pos['position'][0],
                'y_position': pos['position'][1]
            })
    movement_df = pd.DataFrame(movement_data)
    
    os.makedirs('reports', exist_ok=True)
    attendance_df.to_csv('reports/attendance.csv', index=False)
    ppe_df.to_csv('reports/ppe_violations.csv', index=False)
    movement_df.to_csv('reports/movement_patterns.csv', index=False)
    
    print("✅ Reports generated in 'reports/' directory.")

# Main
def main():
    model_path = 'yolov5s.pt'  # or path to PPE-trained YOLOv5 model
    cctv_source = 0  # or video file path
    
    model = initialize_yolov5(model_path)
    monitor_workplace(model, cctv_source)

if __name__ == '__main__':
    main()
