#!/usr/bin/env python3
"""
mosaic_test.py

This script loads four random images (with Pascal VOC–formatted annotations)
from your dataset, applies a simple mosaic augmentation, and visualizes the output.

The mosaic augmentation takes four images and places them into one mosaic image,
adjusting the bounding box coordinates accordingly.

Adjust the folder paths and TARGET_SIZE as needed.
"""

import os
import cv2
import glob
import numpy as np
import random
import matplotlib.pyplot as plt

# ---------------------------
# Global Settings
# ---------------------------
# Change this to match your desired mosaic output size (width, height)
TARGET_SIZE = (1280, 960)

# Change these paths to point to your image and label folders.
IMAGES_FOLDER = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images"
LABELS_FOLDER = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/labels_minmax"

# ---------------------------
# Helper Functions
# ---------------------------
def load_image_and_boxes(img_path, labels_folder):
    """
    Load an image and its corresponding bounding boxes and labels.
    Assumes annotation files have the same basename as the image with a .txt extension,
    and each line in the file is formatted as: class_id xmin ymin xmax ymax
    """
    image = cv2.imread(img_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_folder, base_name + ".txt")
    boxes = []
    labels = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xmin, ymin, xmax, ymax = map(float, parts[1:])
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(cls_id)
    else:
        print(f"Warning: No label file for image {img_path}")
    return image, boxes, labels

def draw_boxes(image, boxes, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on an image.
    Boxes should be in [xmin, ymin, xmax, ymax] format.
    """
    img_copy = image.copy()
    for bbox in boxes:
        x1, y1, x2, y2 = list(map(int, bbox[:4]))
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

# ---------------------------
# Mosaic Augmentation Function
# ---------------------------
def mosaic_augment(images, boxes_list, labels_list, target_size=TARGET_SIZE):
    """
    Create a mosaic image from 4 images.
    
    Parameters:
      - images: list of 4 images (NumPy arrays in RGB)
      - boxes_list: list of bounding boxes for each image (each a list of [xmin, ymin, xmax, ymax])
      - labels_list: list of labels for each image
      - target_size: tuple (width, height) for the mosaic output
    
    Returns:
      - mosaic_img: the mosaic image (RGB, same type as inputs)
      - mosaic_boxes: list of adjusted bounding boxes
      - mosaic_labels: list of corresponding labels
    """
    mosaic_w, mosaic_h = target_size
    # Create a blank canvas; fill with gray (e.g., value 114) as used in some implementations.
    mosaic_img = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)
    
    # Choose a random mosaic center (within a reasonable range)
    xc = random.randint(int(mosaic_w * 0.3), int(mosaic_w * 0.7))
    yc = random.randint(int(mosaic_h * 0.3), int(mosaic_h * 0.7))
    
    mosaic_boxes = []
    mosaic_labels = []
    
    # Coordinates for the 4 placements
    # Order: top-left, top-right, bottom-left, bottom-right.
    placements = [
        (0, 0, xc, yc),                     # top-left
        (xc, 0, mosaic_w, yc),                # top-right
        (0, yc, xc, mosaic_h),                # bottom-left
        (xc, yc, mosaic_w, mosaic_h)          # bottom-right
    ]
    
    for i, (img, boxes, labels) in enumerate(zip(images, boxes_list, labels_list)):
        h, w, _ = img.shape
        x1a, y1a, x2a, y2a = placements[i]
        # Compute region width and height in the mosaic
        region_w = x2a - x1a
        region_h = y2a - y1a
        
        # For simplicity, we resize the image to fill its mosaic quadrant.
        img_resized = cv2.resize(img, (region_w, region_h))
        # Place the resized image in the mosaic.
        mosaic_img[y1a:y2a, x1a:x2a] = img_resized
        
        # Adjust boxes: assume original boxes are relative to the original image.
        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            # Compute scaling factors
            scale_x = region_w / w
            scale_y = region_h / h
            # Scale boxes
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_x
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_y
            # Shift boxes by the mosaic quadrant offset.
            boxes[:, 0] += x1a
            boxes[:, 1] += y1a
            boxes[:, 2] += x1a
            boxes[:, 3] += y1a
            mosaic_boxes.extend(boxes.tolist())
            mosaic_labels.extend(labels)
    
    return mosaic_img, mosaic_boxes, mosaic_labels

# ---------------------------
# Main Routine for Testing Mosaic Augmentation
# ---------------------------
def main():
    # Gather image paths.
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(IMAGES_FOLDER, ext)))
    if len(image_paths) < 4:
        raise ValueError("Need at least 4 images for mosaic augmentation.")
    
    # Randomly select 4 images.
    sample_paths = random.sample(image_paths, 4)
    
    images = []
    boxes_list = []
    labels_list = []
    for path in sample_paths:
        img, boxes, labels = load_image_and_boxes(path, LABELS_FOLDER)
        images.append(img)
        boxes_list.append(boxes)
        labels_list.append(labels)
        print(f"Loaded {os.path.basename(path)} with {len(boxes)} boxes.")
    
    # Apply mosaic augmentation.
    mosaic_img, mosaic_boxes, mosaic_labels = mosaic_augment(images, boxes_list, labels_list, target_size=TARGET_SIZE)
    
    print("\nMosaic Augmentation Result:")
    print(f"Total boxes: {len(mosaic_boxes)}")
    print(f"Labels: {mosaic_labels}")
    
    # Draw bounding boxes on mosaic image.
    mosaic_with_boxes = draw_boxes(mosaic_img, mosaic_boxes, color=(0, 255, 0), thickness=2)
    
    # Display the mosaic image.
    plt.figure(figsize=(12, 8))
    plt.imshow(mosaic_with_boxes)
    plt.title("Mosaic Augmented Image with Boxes")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
