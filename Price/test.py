import cv2
from pyzbar.pyzbar import decode
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import sqlite3
import pyrealsense2 as rs
import numpy as np

# Connect to SQLite database
db_path = 'products1.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Initialize the OCR model
model = ocr_predictor(pretrained=True)

# Configure RealSense camera pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB stream
pipeline.start(config)

try:
    while True:
        # Capture a frame from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # Convert RealSense frame to a numpy array (OpenCV image)
        image = np.asanyarray(color_frame.get_data())

        # Decode the barcodes from the frame
        barcodes = decode(image)
        left_bottom_coordinates = []

        for idx, barcode in enumerate(barcodes, start=1):
            barcode_data = barcode.data.decode('utf-8')
            print(f"Barcode {idx}: {barcode_data}")

            x, y, w, h = barcode.rect
            left_bottom_coord = (x, y + h)
            left_bottom_coordinates.append(left_bottom_coord)
            print(f"Left bottom coordinates of Barcode {idx}: {left_bottom_coord}")

            margin = 340
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(image.shape[1], x + w + margin)
            y_end = min(image.shape[0], y + h + margin)
            expanded_region = image[y_start:y_end, x_start:x_end]

            # Perform OCR on the cropped expanded region
            doc = DocumentFile.from_images(expanded_region)
            result = model(doc)
            extracted_text = result.render()

            # Extract product name and price
            lines = extracted_text.splitlines()
            product_name = ""
            price = ""
            for line in lines:
                if "$" in line:
                    price = line.strip()
                else:
                    product_name = line.strip()

            if not product_name or not price:
                print(f"Skipping Barcode {barcode_data}: Missing product name or price")
                continue

            # Correct price formatting
            if price.startswith("$"):
                raw_price = price[1:].replace(",", "")
                if raw_price.isdigit() and len(raw_price) > 2:
                    corrected_price = f"${raw_price[:-2]}.{raw_price[-2:]}"

                else:
                    corrected_price = price
            else:
                corrected_price = price

            # Search for the barcode in the database
            cursor.execute("SELECT price FROM products WHERE barcode=?", (barcode_data,))
            db_result = cursor.fetchone()
            if db_result:
                db_price = db_result[0]
                if corrected_price == db_price:
                    print(f"Price is the same for Barcode {barcode_data}: {corrected_price}")
                else:
                    print(f"Price mismatch for Barcode {barcode_data}: OCR Price = {corrected_price}, Database Price = {db_price}")
            else:
                print(f"No price found in the database for Barcode {barcode_data}")
            print(f"Product Name: {product_name}")
            print(f"OCR Extracted Price: {corrected_price}")
            print("-" * 40)

        # Additional code to process left_bottom_coordinates and draw rectangles
        print("All Left Bottom Coordinates:", left_bottom_coordinates)
        x_cord = []
        for x in left_bottom_coordinates:
            for y in x:
                x_cord.append(y)
                break
        x_cord.sort()
        max_margin = x_cord[0]
        print(max_margin)
        max_margin = x_cord[0] + 150
        y_axis = [(0, 0)]
        for x in left_bottom_coordinates:
            for y in x:
                if y <= max_margin:
                    y_axis.append(x)
        # sort the list of tuples by the second element
        y_axis.sort(key=lambda x: x[1])
        print(y_axis)
        # Create a new list with the updated second values (excluding the first tuple)
        updated_coordinates = [(x, y) if i == 0 else (x, y + 200) for i, (x, y) in enumerate(y_axis)]
        # Print the updated list
        print(updated_coordinates)

        # Proceed to draw rectangles based on barcode coordinates
        for i in range(len(updated_coordinates) - 1):
            # Get the y-coordinates for the horizontal box
            y_start = updated_coordinates[i][1]
            y_end = updated_coordinates[i + 1][1]
            
            # Find all barcodes within this horizontal range based on their y-coordinates
            relevant_barcodes = [coord for coord in left_bottom_coordinates if y_start <= coord[1] <= y_end]
            
            # Sort barcodes by the x-coordinate (left to right)
            relevant_barcodes.sort(key=lambda x: x[0])
            
            # Now, create boxes around each barcode
            for j in range(len(relevant_barcodes) - 1):
                # Get the x-coordinates for the vertical box
                x_start = relevant_barcodes[j][0]
                x_end = relevant_barcodes[j + 1][0]

                # Draw a rectangle around the barcode (top-left and bottom-right corners)
                top_left = (x_start, y_start)
                bottom_right = (x_end, y_end)
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)  # Green box with 2px thickness
                
                print(f"Drew rectangle {j + 1} of horizontal slice {i + 1}: {top_left} to {bottom_right}")
            
            # Handle the last segment from the last barcode to the right edge of the image
            if relevant_barcodes:
                x_last_start = relevant_barcodes[-1][0]
                x_last_end = image.shape[1]  # Right edge of the image
                
                # Draw a rectangle for the remaining right part
                top_left = (x_last_start, y_start)
                bottom_right = (x_last_end, y_end)
                cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
                
                print(f"Drew last rectangle of horizontal slice {i + 1}: {top_left} to {bottom_right}")

        # Display the image with boxes
        cv2.imshow("Barcode Detection", image)

        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop RealSense pipeline and close database connection
    pipeline.stop()
    conn.close()
    cv2.destroyAllWindows()
