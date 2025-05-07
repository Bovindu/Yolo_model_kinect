import cv2
import zmq
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # Auto-downloads if missing

# ZeroMQ setup
context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

print("YOLO server ready. Waiting for Kinect frames...")
while True:
    # Receive frame from C#
    frame_data = socket.recv()
    np_arr = np.frombuffer(frame_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Run YOLO inference
    results = model.predict(frame, imgsz=640, conf=0.5)

    # Format detections as JSON
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert to integers
        detections.append({
            "label": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": [x1, y1, x2, y2]  # Now integers
        })

    # Send back to C#
    socket.send_json(detections)