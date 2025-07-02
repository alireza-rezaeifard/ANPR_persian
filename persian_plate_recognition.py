#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras import models

# Limit TensorFlow to CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

class PersianPlateRecognition:
    def __init__(self):
        self.plate_detector = None
        self.ocr_model = None
        self.class_names = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
            'A', 'B', 'D', 'Gh', 'H', 'J', 'L', 'M', 'N', 'P', 'PuV', 'PwD', 'Sad', 'Sin', 'T', 'Taxi', 'V', 'Y'  # Persian characters
        ]
        
    def load_models(self, plate_model_path, ocr_model_path):
        """
        Load the trained license plate detection and OCR models
        """
        print("Loading models...")
        
        # Load the license plate detector
        if os.path.exists(plate_model_path):
            self.plate_detector = YOLO(plate_model_path)
            print(f"License plate detector loaded from {plate_model_path}")
        else:
            print(f"Error: License plate model not found at {plate_model_path}")
            
        # Load the OCR model
        if os.path.exists(ocr_model_path):
            self.ocr_model = models.load_model(ocr_model_path)
            print(f"OCR model loaded from {ocr_model_path}")
        else:
            print(f"Error: OCR model not found at {ocr_model_path}")
            
        return self.plate_detector is not None and self.ocr_model is not None
    
    def preprocess_plate(self, plate_img):
        """
        Preprocess the license plate image for OCR
        """
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY) if len(plate_img.shape) == 3 else plate_img
        
        # Resize to a fixed size
        resized = cv2.resize(gray, (640, 128))
        
        # Apply adaptive thresholding for better character segmentation
        thresh = cv2.adaptiveThreshold(resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY_INV, 11, 2)
        
        # Dilate to connect broken parts of characters
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        
        # Find contours (characters)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from right to left (for Persian reading direction)
        contours = sorted(contours, key=lambda cnt: cv2.boundingRect(cnt)[0], reverse=True)
        
        # Extract and collect character images
        char_images = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small contours (noise)
            if w > 10 and h > 20:
                char_img = thresh[y:y+h, x:x+w]
                
                # Resize to match OCR model input
                char_img = cv2.resize(char_img, (64, 64))
                
                # Add padding if needed
                if char_img.shape[0] != 64 or char_img.shape[1] != 64:
                    char_img = cv2.resize(char_img, (64, 64))
                
                char_images.append(char_img)
        
        return char_images
    
    def recognize_characters(self, char_images):
        """
        Recognize characters in the extracted character images
        """
        if self.ocr_model is None:
            print("Error: OCR model not loaded")
            return ""
        
        plate_text = ""
        
        for char_img in char_images:
            # Prepare the image for the model
            img = np.expand_dims(char_img / 255.0, axis=-1)  # Add channel dimension
            img = np.expand_dims(img, axis=0)  # Add batch dimension
            
            # Predict
            predictions = self.ocr_model.predict(img, verbose=0)
            char_idx = np.argmax(predictions[0])
            
            # Get the character
            if char_idx < len(self.class_names):
                plate_text += self.class_names[char_idx]
        
        return plate_text
    
    def process_image(self, image_path, conf_threshold=0.5):
        """
        Process an image to detect and recognize license plates
        """
        if self.plate_detector is None:
            print("Error: Plate detector model not loaded")
            return None, []
        
        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not read image from {image_path}")
            return None, []
        
        # Detect license plates
        results = self.plate_detector(image, conf=conf_threshold, device='cpu')
        
        plates_text = []
        plates_imgs = []
        
        # Process each detected license plate
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            
            for box in boxes:
                x1, y1, x2, y2, score, class_id = box
                
                if score > conf_threshold:
                    # Extract the license plate region
                    plate_img = image[int(y1):int(y2), int(x1):int(x2)]
                    plates_imgs.append(plate_img)
                    
                    # Preprocess the plate and extract characters
                    char_images = self.preprocess_plate(plate_img)
                    
                    # Recognize the characters
                    plate_text = self.recognize_characters(char_images)
                    plates_text.append(plate_text)
                    
                    # Draw bounding box and text on the image
                    cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(image, plate_text, (int(x1), int(y1) - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return image, plates_text

def main():
    # Initialize the Persian plate recognition system
    plate_recognition = PersianPlateRecognition()
    
    # Load models
    plate_model_path = 'runs/detect/license_plate_detector/weights/best.pt'
    ocr_model_path = 'persian_ocr_model.h5'
    
    models_loaded = plate_recognition.load_models(plate_model_path, ocr_model_path)
    
    if not models_loaded:
        print("Error: Failed to load one or more models. Please train the models first.")
        print("Run 'python train_license_plate.py' to train the plate detector.")
        print("Run 'python train_persian_ocr.py' to train the OCR model.")
        return
    
    # Process test images
    test_dir = 'images/test'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.jpg')][:5]  # Process first 5 images
    
    print(f"Processing {len(test_images)} test images...")
    
    for image_name in test_images:
        image_path = os.path.join(test_dir, image_name)
        
        # Process the image
        output_image, plates_text = plate_recognition.process_image(image_path)
        
        if output_image is not None:
            # Save the output image
            output_path = os.path.join(output_dir, f"detected_{image_name}")
            cv2.imwrite(output_path, output_image)
            
            # Print the recognized license plates
            print(f"Image: {image_name}")
            for i, plate in enumerate(plates_text):
                print(f"  Plate {i+1}: {plate}")
    
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main() 