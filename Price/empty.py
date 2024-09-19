import pyrealsense2 as rs
import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient
import requests
import base64

# Initialize the RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start the pipeline
pipeline.start(config)

# Initialize InferenceHTTPClient
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="1gcPa1rF1UMHGBEVWtfp"
)

# Function to encode image for API
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    jpg_as_text = base64.b64encode(buffer).decode('utf-8')
    return jpg_as_text

try:
    while True:
        # Capture frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert images to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Send frame for inference
        encoded_frame = encode_image(frame)
        response = CLIENT.infer(encoded_frame, model_id="empty-shelf-detector/1")

        # Display the result and frame
        result = response.get('predictions', [])
        print("Inference result:", result)

        # Show the camera feed with detection results
        cv2.imshow('RealSense', frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
