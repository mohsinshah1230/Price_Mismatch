import cv2
import pyrealsense2 as rs
import torch
from ultralytics import YOLO
import numpy as np
from roboflow import Roboflow

# Initialize Roboflow and get the model for empty shelf detection
rf = Roboflow(api_key="1gcPa1rF1UMHGBEVWtfp")
project = rf.workspace().project("empty-shelf-detector")
model7 = project.version(1).model  # Save model in model7

# Load the models
model1 = YOLO('best1.pt')  # Product Detection
model5 = YOLO('best5.pt')  # Product Detection
model6 = YOLO('best.pt')   # Product Detection

# Initialize the RealSense pipeline for streaming
pipeline = rs.pipeline()

# Configure the streams (color and depth)
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth stream
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # Color stream

# Start streaming
pipeline.start(config)

try:
    while True:
        # Wait for a coherent pair of frames (depth and color)
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert the color image to a numpy array
        color_image = np.asanyarray(color_frame.get_data())

        # Run detection on YOLO models and Roboflow model
        results1 = model1(color_image)  # Product detection
        results5 = model5(color_image)  # Product detection
        results6 = model6(color_image)  # Product detection
        results7 = model7.predict(color_image)  # Empty shelf detection using Roboflow

        # Combine all results into one list (or process them separately if needed)
        all_results = [results1, results5, results6, results7]

        # Loop through all results and display the detections
        for results in all_results:
            for result in results:
                # Extract bounding box coordinates and labels
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    label = int(box.cls.item())  # Class label (convert to int)
                    conf = float(box.conf.item())  # Confidence score (convert to float)

                    # Draw bounding box and label on the image
                    cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(color_image, f'{label} {conf:.2f}', (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the image with detections
        cv2.imshow('RealSense - YOLO and Roboflow Detections', color_image)

        # Break the loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close all windows
    pipeline.stop()
    cv2.destroyAllWindows()
