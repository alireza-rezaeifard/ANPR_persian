#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
from ultralytics import YOLO

def train_plate_detection_model():
    """
    Train a YOLOv8 model for license plate detection
    """
    print("Starting license plate detection model training...")
    
    # Check data.yaml file
    with open('data.yaml', 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)
        print(f"Training with {len(data_config['names'])} classes: {data_config['names']}")
    
    # Start with a pretrained YOLOv8 model
    model = YOLO('yolov8s.pt')
    
    # Train the model with our custom dataset
    results = model.train(
        data='data.yaml',
        epochs=50,
        imgsz=640,
        batch=8,  # Reduced batch size for CPU training
        name='license_plate_detector',
        verbose=True,
        device='cpu',  # Use CPU instead of GPU
        patience=10,  # Early stopping patience
        save=True    # Save best model
    )
    
    print(f"Training completed. Best model saved at: {results.best}")
    
    return results

def validate_plate_detection_model():
    """
    Validate the trained license plate detection model
    """
    print("Validating license plate detection model...")
    
    # Load the best trained model
    model_path = 'runs/detect/license_plate_detector/weights/best.pt'
    if not os.path.exists(model_path):
        print(f"Error: Trained model not found at {model_path}")
        return
    
    model = YOLO(model_path)
    
    # Validate the model
    results = model.val(data='data.yaml', device='cpu')
    
    print(f"Validation metrics: {results.box.map}")
    
    return results

if __name__ == "__main__":
    # Train the license plate detection model
    train_results = train_plate_detection_model()
    
    # Validate the trained model
    val_results = validate_plate_detection_model()
    
    print("License plate detection model training complete!")