import cv2
from pyzbar.pyzbar import decode
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import sqlite3
from ultralytics import YOLO
import pyrealsense2 as rs

# Connect to SQLite database
db_path = 'products1.db'
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Initialize the OCR model
model = ocr_predictor(pretrained=True)

while True:
    # Wait for a coherent frame
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        continue

    # Convert RealSense frame to OpenCV format
    image = np.asanyarray(color_frame.get_data())

    # Decode the barcodes
    barcodes = decode(image)
    # List to store the left bottom coordinates of each barcode
    left_bottom_coordinates = []

    # Process each barcode
    for idx, barcode in enumerate(barcodes, start=1):
        # Decode the barcode data
        barcode_data = barcode.data.decode('utf-8')
        print(f"Barcode {idx}: {barcode_data}")
        # Get the region of the barcode
        x, y, w, h = barcode.rect
        # Calculate the left bottom coordinates (x, y + h)
        left_bottom_coord = (x, y + h)
        left_bottom_coordinates.append(left_bottom_coord)  # Save the coordinates to the list
        print(f"Left bottom coordinates of Barcode {idx}: {left_bottom_coord}")
        # Expand the crop area to include surrounding content (increase margins)
        margin = 340  # You can adjust the margin size
        x_start = max(0, x - margin)
        y_start = max(0, y - margin)
        x_end = min(image.shape[1], x + w + margin)
        y_end = min(image.shape[0], y + h + margin)
        # Crop the expanded region
        expanded_region = image[y_start:y_end, x_start:x_end]
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

    # Print all left bottom coordinates
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
    y_axis = [(0,0)]
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

    # Load the models
    model2 = YOLO('best2.pt')  # Empty Shelf Detector
    model3 = YOLO('best3.pt')  # Low Inventory Detector
    model4 = YOLO('best4.pt')  # Low Inventory Detector
    model5 = YOLO('best5.pt')  # Product Detection
    model6 = YOLO('best.pt')   # Product Detection

    # Iterate through the coordinates
    for i in range(len(updated_coordinates) - 1):
        y_start = updated_coordinates[i][1]
        y_end = updated_coordinates[i + 1][1]

        # Find all barcodes within this horizontal range based on their y-coordinates
        relevant_barcodes = [coord for coord in left_bottom_coordinates if y_start <= coord[1] <= y_end]

        # Sort barcodes by the x-coordinate (left to right)
        relevant_barcodes.sort(key=lambda x: x[0])

        for j in range(len(relevant_barcodes) - 1):
            x_start = relevant_barcodes[j][0]
            x_end = relevant_barcodes[j + 1][0]

            # Draw rectangle around the barcode
            top_left = (x_start, y_start)
            bottom_right = (x_end, y_end)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            print(f"Drew rectangle {j + 1} of horizontal slice {i + 1}: {top_left} to {bottom_right}")

            # Crop the image inside the box
            cropped_box = image[y_start:y_end, x_start:x_end]

            # Detect objects inside the cropped box using the YOLO models
            result_empty_shelf = model2(cropped_box)
            result_low_inventory = model3(cropped_box)
            result_product = model5(cropped_box)

            # Initialize detection results
            box_result = ""

            # Check for empty shelf
            if result_empty_shelf[0].boxes:
                box_result = "Empty Shelf"
            
            # Check for low inventory
            elif result_low_inventory[0].boxes:
                box_result = "Low Inventory Detected"

            # Check for product detection
            elif result_product[0].boxes:
                detected_product_class = result_product[0].names[result_product[0].boxes[0].cls]
                box_result = f"Product Detected: {detected_product_class}"

            else:
                box_result = "No Product Detected"

            # Output the result for this box
            print(f"Box {j + 1} of slice {i + 1}: {box_result}")

        # Handle the last segment to the right edge of the image
        if relevant_barcodes:
            x_last_start = relevant_barcodes[-1][0]
            x_last_end = image.shape[1]

            top_left = (x_last_start, y_start)
            bottom_right = (x_last_end, y_end)
            cv2.rectangle(image, top_left, bottom_right, (0, 255, 0), 2)
            print(f"Drew last rectangle of horizontal slice {i + 1}: {top_left} to {bottom_right}")

            # Crop the last section
            cropped_box = image[y_start:y_end, x_last_start:x_last_end]

            # Detect objects in the cropped box
            result_empty_shelf = model2(cropped_box)
            result_product = model5(cropped_box)
            result_product1 = model6(cropped_box)

            # Initialize detection results
            box_result = ""

            # Check for empty shelf
            if result_empty_shelf[0].boxes:
                box_result = "Empty Shelf"
            
            # Check for low inventory
            elif result_low_inventory[0].boxes:
                box_result = "Low Inventory Detected"

            # Check for product detection
            elif result_product[0].boxes:
                detected_product_class = result_product[0].names[result_product[0].boxes[0].cls]
                box_result = f"Product Detected: {detected_product_class}"

            # Check for product detection
            elif result_product1[0].boxes:
                detected_product_class = result_product1[0].names[result_product1[0].boxes[0].cls]
                box_result = f"Product Detected: {detected_product_class}"

            else:
                box_result = "No Product Detected"

            # Output the result for this box
            print(f"Box last of slice {i + 1}: {box_result}")

    # Show the image with detected barcodes
    cv2.imshow('Detected Barcodes', image)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Stop the RealSense pipeline
pipeline.stop()

# Release database connection
conn.close()
