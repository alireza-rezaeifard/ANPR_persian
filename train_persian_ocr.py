#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Limit TensorFlow to CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def create_ocr_model(input_shape, num_classes):
    """
    Create a CNN-LSTM model with CTC loss for OCR
    """
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # CNN layers for feature extraction
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Prepare feature maps for LSTM
    new_shape = ((input_shape[0] // 4), (input_shape[1] // 4) * 128)
    x = layers.Reshape(new_shape)(x)
    
    # Bidirectional LSTM layers
    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    
    # Output layer
    outputs = layers.Dense(num_classes + 1, activation='softmax')(x)  # +1 for blank character in CTC
    
    # Create and compile model
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # CTC loss function
    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return tf.keras.backend.ctc_batch_cost(labels, y_pred, input_length, label_length)
    
    # Define additional inputs for CTC loss
    labels = layers.Input(name='labels', shape=[None], dtype='float32')
    input_length = layers.Input(name='input_length', shape=[1], dtype='int64')
    label_length = layers.Input(name='label_length', shape=[1], dtype='int64')
    
    # Define the CTC loss
    loss_out = layers.Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')(
        [outputs, labels, input_length, label_length])
    
    # Create and compile training model
    train_model = models.Model(inputs=[inputs, labels, input_length, label_length], 
                            outputs=loss_out)
    train_model.compile(optimizer='adam', loss={'ctc': lambda y_true, y_pred: y_pred})
    
    return train_model, model

def prepare_iranis_dataset():
    """
    Prepare the Iranis dataset for training
    """
    print("Preparing Iranis dataset for OCR training...")
    
    # Path to the Iranis dataset
    dataset_path = 'Iranis-dataset-master/Iranis Dataset Files'
    
    # Character classes in the dataset
    classes = [
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',  # Digits
        'A', 'B', 'D', 'Gh', 'H', 'J', 'L', 'M', 'N', 'P', 'PuV', 'PwD', 'Sad', 'Sin', 'T', 'Taxi', 'V', 'Y'  # Persian characters
    ]
    
    # Check if the directory exists
    if not os.path.exists(dataset_path):
        print(f"Error: Iranis dataset directory not found at {dataset_path}")
        return None
    
    # Create directories for processed data
    os.makedirs('ocr_dataset/train', exist_ok=True)
    os.makedirs('ocr_dataset/valid', exist_ok=True)
    os.makedirs('ocr_dataset/test', exist_ok=True)
    
    # Create class directories
    for class_name in classes:
        os.makedirs(f'ocr_dataset/train/{class_name}', exist_ok=True)
        os.makedirs(f'ocr_dataset/valid/{class_name}', exist_ok=True)
        os.makedirs(f'ocr_dataset/test/{class_name}', exist_ok=True)
    
    # Process and split each class
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_name} not found")
            continue
        
        image_files = [f for f in os.listdir(class_path) if f.endswith('.jpg')]
        
        # Split into train (70%), validation (15%), and test (15%)
        num_samples = len(image_files)
        num_train = int(0.7 * num_samples)
        num_val = int(0.15 * num_samples)
        
        # Shuffle files
        np.random.shuffle(image_files)
        
        train_files = image_files[:num_train]
        val_files = image_files[num_train:num_train+num_val]
        test_files = image_files[num_train+num_val:]
        
        # Process training files
        for idx, file_name in enumerate(train_files):
            try:
                src_path = os.path.join(class_path, file_name)
                dst_path = os.path.join(f'ocr_dataset/train/{class_name}', f"{class_name}_{idx}.jpg")
                
                # Read, resize and save the image
                img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    cv2.imwrite(dst_path, img)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        
        # Process validation files
        for idx, file_name in enumerate(val_files):
            try:
                src_path = os.path.join(class_path, file_name)
                dst_path = os.path.join(f'ocr_dataset/valid/{class_name}', f"{class_name}_{idx}.jpg")
                
                # Read, resize and save the image
                img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    cv2.imwrite(dst_path, img)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
        
        # Process test files
        for idx, file_name in enumerate(test_files):
            try:
                src_path = os.path.join(class_path, file_name)
                dst_path = os.path.join(f'ocr_dataset/test/{class_name}', f"{class_name}_{idx}.jpg")
                
                # Read, resize and save the image
                img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (64, 64))
                    cv2.imwrite(dst_path, img)
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
                
        print(f"Processed {class_name} class: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
    
    return classes

def train_persian_ocr(classes):
    """
    Train a Persian OCR model using the prepared Iranis dataset
    """
    print("Training Persian OCR model...")
    
    num_classes = len(classes)
    input_shape = (64, 64, 1)  # Height, width, channels
    
    # Create the model
    train_model, prediction_model = create_ocr_model(input_shape, num_classes)
    
    # Use a smaller batch size for CPU training
    batch_size = 16
    
    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1
    )
    
    validation_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        'ocr_dataset/train',
        target_size=(64, 64),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = validation_datagen.flow_from_directory(
        'ocr_dataset/valid',
        target_size=(64, 64),
        batch_size=batch_size,
        color_mode='grayscale',
        class_mode='categorical',
        shuffle=False
    )
    
    # Callbacks for training
    checkpoint = ModelCheckpoint(
        'persian_ocr_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.0001,
        verbose=1
    )
    
    # Train the model
    history = prediction_model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=30,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint, early_stopping, reduce_lr]
    )
    
    # Save the model
    prediction_model.save('persian_ocr_model.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train')
    plt.plot(history.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('persian_ocr_training.png')
    
    print("Persian OCR model training completed!")
    
    return prediction_model

if __name__ == "__main__":
    # Prepare the dataset
    classes = prepare_iranis_dataset()
    
    if classes:
        # Train the OCR model
        model = train_persian_ocr(classes)
        
        print("Persian OCR training complete!")