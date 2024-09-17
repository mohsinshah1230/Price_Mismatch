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

        # Decode the barcodes
        barcodes = decode(color_image)

        # Process each barcode
        for idx, barcode in enumerate(barcodes, start=1):
            # Decode the barcode data
            barcode_data = barcode.data.decode('utf-8')
            print(f"Barcode {idx}: {barcode_data}")
            
            # Get the region of the barcode
            x, y, w, h = barcode.rect

            # Expand the crop area to include surrounding content (increase margins)
            margin = 340  # Adjust the margin size if needed
            x_start = max(0, x - margin)
            y_start = max(0, y - margin)
            x_end = min(color_image.shape[1], x + w + margin)
            y_end = min(color_image.shape[0], y + h + margin)

            # Crop the expanded region
            expanded_region = color_image[y_start:y_end, x_start:x_end]

            # Save the cropped region to a temporary image file for OCR processing
            temp_image_path = f"barcode_expanded_{idx}.jpg"
            cv2.imwrite(temp_image_path, expanded_region)

            # Perform OCR on the cropped expanded region
            doc = DocumentFile.from_images(temp_image_path)
            result = model(doc)

            # Convert the OCR result to a string
            extracted_text = result.render()

            # Extract product name and price from the OCR result
            lines = extracted_text.splitlines()
            product_name = ""
            price = ""

            for line in lines:
                if "$" in line:
                    price = line.strip()
                else:
                    product_name = line.strip()

            # Skip the barcode if either product name or price is not found
            if not product_name or not price:
                print(f"Skipping Barcode {barcode_data}: Missing product name or price")
                continue

            # Correct the price formatting
            if price and price.startswith("$"):
                # Remove the dollar sign for further processing
                raw_price = price[1:].replace(",", "")
                
                # If the price is numeric and doesn't contain a decimal, add one
                if raw_price.isdigit() and len(raw_price) > 2:
                    # Insert the decimal point two places from the end
                    corrected_price = f"${raw_price[:-2]}.{raw_price[-2:]}"
                else:
                    corrected_price = price  # Use the extracted price if it's already correct
            else:
                corrected_price = price

            # Search for the barcode in the database and fetch the price
            cursor.execute("SELECT price FROM products WHERE barcode=?", (barcode_data,))
            db_result = cursor.fetchone()

            if db_result:
                db_price = db_result[0]
                # Compare the extracted price with the database price
                if corrected_price == db_price:
                    print(f"Price is the same for Barcode {barcode_data}: {corrected_price}")
                else:
                    print(f"Price mismatch for Barcode {barcode_data}: OCR Price = {corrected_price}, Database Price = {db_price}")
            else:
                print(f"No price found in the database for Barcode {barcode_data}")

            print(f"Product Name: {product_name}")
            print(f"OCR Extracted Price: {corrected_price}")
            print("-" * 40)  # Separator for better readability

        # Display the color image
        cv2.imshow('RealSense D435 - Color', color_image)

        # Break the loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Stop the pipeline and close all windows
    pipeline.stop()
    cv2.destroyAllWindows()

# Close the database connection
conn.close()
