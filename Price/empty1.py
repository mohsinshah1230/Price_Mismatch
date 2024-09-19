import pyrealsense2 as rs
import cv2
from roboflow import Roboflow
import supervision as sv

# Step 1: Initialize and save the model in model7
rf = Roboflow(api_key="1gcPa1rF1UMHGBEVWtfp")
project = rf.workspace().project("empty-shelf-detector")
model7 = project.version(1).model  # Model saved in model7

# Step 2: Initialize RealSense camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Step 3: Set up annotators
label_annotator = sv.LabelAnnotator()
box_annotator = sv.BoxAnnotator()

try:
    while True:
        # Capture frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        if not color_frame:
            continue

        # Convert RealSense frame to a NumPy array
        frame = np.asanyarray(color_frame.get_data())

        # Step 4: Use model7 for inference on the captured frame
        _, buffer = cv2.imencode('.jpg', frame)
        result = model7.predict(buffer.tobytes(), confidence=40, overlap=30).json()

        # Process predictions
        labels = [item["class"] for item in result["predictions"]]
        detections = sv.Detections.from_roboflow(result)

        # Annotate the image with bounding boxes and labels
        annotated_image = box_annotator.annotate(scene=frame, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # Step 5: Display the annotated RealSense stream
        cv2.imshow('RealSense Viewer', annotated_image)

        # Press 'q' to exit the viewer
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the RealSense pipeline
    pipeline.stop()
    cv2.destroyAllWindows()
