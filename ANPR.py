import cv2 as cv
from ultralytics import YOLO
import numpy as np
import os

# Use a proper license plate detection model
# Download a pre-trained license plate model (example):
# model = YOLO('license_plate_best.pt')  # Replace with your actual model path

# For testing with standard YOLO (but this won't detect license plates!)
# Use a custom-trained license plate model for actual use
model = YOLO('yolov8n.pt')  # Not ideal for license plates - just for testing

image_path = '00001.jpg'

# Verify image exists
if not os.path.exists(image_path):
    print(f"Error: Image not found at {image_path}")
    exit(1)

image = cv.imread(image_path)
if image is None:
    print("Error: Failed to load image")
    exit(1)

# Run detection with lower confidence threshold
results = model(image, conf=0.2)  # Reduced from 0.5

# Print debug information
print(f"Number of results: {len(results)}")
print(f"Model classes: {model.names}")

plate_found = False
plate_class_id = 0  # This may vary depending on your model

for result in results:
    print(f"Boxes in result: {result.boxes.shape[0]}")

    for bbox in result.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = bbox

        # Print all detected objects for debugging
        print(f"Detected class {class_id} with score {score:.2f}")

        # If using standard YOLO, look for car class (2) as proxy
        if score > 0.2:  # Even lower threshold for debugging
            plate_found = True
            cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Extract and save plate region
            plate_image = image[int(y1):int(y2), int(x1):int(x2)]
            cv.imwrite('plate.png', plate_image)

            # Display results
            cv.imshow('Detection Result', image)
            cv.imshow('Detected Plate', plate_image)

            # Print details about the detection
            print(f"\nDetected object:")
            print(f"Class ID: {class_id}")
            print(f"Score: {score:.2f}")
            print(f"Coordinates: ({int(x1)}, {int(y1)}) to ({int(x2)}, {int(y2)})")

if not plate_found:
    print("\nNo license plates detected")
    print("Try:")
    print("1. Using a proper license plate detection model")
    print("2. Checking your image quality")
    print("3. Adjusting confidence threshold")
    print("4. Verifying class ID for license plates in your model")

cv.waitKey(0)
cv.destroyAllWindows()