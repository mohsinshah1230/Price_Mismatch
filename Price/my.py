import cv2

# Open a connection to the camera (usually camera index 0 for default camera)
camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set the resolution to 4K (3840x2160)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# Check if the camera opened successfully
if not camera.isOpened():
    print("Error: Could not open the camera.")
    exit()

print("Camera opened successfully. Press 'q' to exit.")

while True:
    # Capture frame-by-frame
    ret, frame = camera.read()

    # If frame was not grabbed, break the loop
    if not ret:
        print("Failed to grab frame.")
        break

    # Display the resulting frame
    cv2.imshow('4K Camera Feed', frame)

    # Exit when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
camera.release()
cv2.destroyAllWindows()
