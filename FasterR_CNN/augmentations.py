#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
import albumentations as A
import random
import matplotlib.pyplot as plt

##################################
# SINGLE PLACE TO CHANGE IMAGE SIZE
##################################
TARGET_SIZE = (1280, 960)  # (width, height)

##################################
# DebugWrapper: Wraps an Albumentations transform.
##################################
class DebugWrapper:
    """
    Wraps an Albumentations transform to optionally print debug information.
    Currently the internal prints are commented out for clarity.
    """
    def __init__(self, transform, label=""):
        self.transform = transform
        self.label = label

    def __call__(self, **kwargs):
        # Commented out debug prints:
        # image = kwargs.get("image")
        # bboxes = kwargs.get("bboxes", [])
        # print(f"\nDEBUG [{self.label}] - BEFORE transformation")
        # print("  Image shape:", image.shape if image is not None else "None")
        # print("  BBoxes before:", bboxes)
        
        result = self.transform(**kwargs)
        
        # result_image = result.get("image")
        # result_bboxes = result.get("bboxes", [])
        # print(f"DEBUG [{self.label}] - AFTER transformation")
        # print("  Image shape:", result_image.shape if result_image is not None else "None")
        # print("  BBoxes after:", result_bboxes)
        return result

    def __getattr__(self, name):
        return getattr(self.transform, name)

##################################
# count_altered_boxes: Compare original vs. augmented boxes.
##################################
def count_altered_boxes(original_boxes, augmented_boxes, tol=1e-3):
    count = 0
    n = min(len(original_boxes), len(augmented_boxes))
    for i in range(n):
        orig = original_boxes[i]
        aug = augmented_boxes[i]
        if any(abs(o - a) > tol for o, a in zip(orig, aug)):
            count += 1
    return count

##################################
# YOLO-style Pipeline using two Affine transforms
##################################
def get_train_transforms():
    transforms_list = []
    
    # 1. Horizontal flip (p=0.5)
    transforms_list.append(A.HorizontalFlip(p=0.5))
    
    # 2. HSV augmentation (p=0.5)
    transforms_list.append(A.HueSaturationValue(
        hue_shift_limit=10,  # ±10 degrees
        sat_shift_limit=30,  # ±30 (out of 255)
        val_shift_limit=30,  # ±30 (out of 255)
        p=0.5
    ))
    
    # 3. Affine #1: Translate only.
    transforms_list.append(DebugWrapper(
        A.Affine(
            translate_percent={"x": (0.0, 1.0), "y": (0.0, 1.0)},
            rotate=0,
            shear=0,
            scale=1.0,
            p=0.1
        ),
        label="Affine_TranslateOnly"
    ))
    
    # 4. Affine #2: Scale only.
    transforms_list.append(DebugWrapper(
        A.Affine(
            translate_percent=0,
            rotate=0,
            shear=0,
            scale=(0.1, 0.5),
            p=0.5
        ),
        label="Affine_ScaleOnly"
    ))
    
    # 5. Random erasing via CoarseDropout (p=0.4)
    transforms_list.append(A.CoarseDropout(
    num_holes_range=(1, 8),
    hole_height_range=(64, 64),
    hole_width_range=(64, 64),
    fill=0,
    p=0.4
    ))

    
    # 6. Final resize to TARGET_SIZE.
    transforms_list.append(A.Resize(
        width=TARGET_SIZE[0],
        height=TARGET_SIZE[1],
        interpolation=cv2.INTER_LINEAR,
        p=1.0
    ))
    
    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids'],
            min_visibility=0.2
        )
    )

##################################
# RandAugment-like Pipeline using SomeOf inside OneOf
##################################
def get_randaugment_pipeline():
    some_of = A.SomeOf([
         A.Rotate(limit=30, p=1.0),
         A.Affine(translate_percent={"x": (-0.1, 0.1)}, rotate=0, shear=0, scale=1.0, p=1.0),
         A.Affine(translate_percent={"y": (-0.1, 0.1)}, rotate=0, shear=0, scale=1.0, p=1.0),
         A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=1.0),
         A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
         A.RandomBrightnessContrast(p=1.0)
    ], n=2, replace=False, p=0.5)
    
    transforms_list = [DebugWrapper(some_of, label="RandAugment_SomeOf")]
    
    transforms_list.append(A.Resize(
        width=TARGET_SIZE[0],
        height=TARGET_SIZE[1],
        interpolation=cv2.INTER_LINEAR,
        p=1.0
    ))
    
    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(
            format='pascal_voc',
            label_fields=['category_ids'],
            min_visibility=0.2
        )
    )

def mixup_augment(image1, bboxes1, cat_ids1,
                  image2, bboxes2, cat_ids2,
                  mixup_prob=0.0, alpha=32):
    if random.random() > mixup_prob:
        return image1, bboxes1, cat_ids1

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    ratio = np.random.beta(alpha, alpha)
    mixed_img = (ratio * image1 + (1 - ratio) * image2).astype(image1.dtype)
    mixed_bboxes = bboxes1 + bboxes2
    mixed_cat_ids = cat_ids1 + cat_ids2
    return mixed_img, mixed_bboxes, mixed_cat_ids

def load_image_and_boxes(img_path, labels_folder):
    image_bgr = cv2.imread(img_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(labels_folder, base_name + ".txt")
    bboxes = []
    category_ids = []
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls_id = int(parts[0])
                    xmin, ymin, xmax, ymax = map(float, parts[1:])
                    bboxes.append([xmin, ymin, xmax, ymax])
                    category_ids.append(cls_id)
    else:
        print(f"Warning: No label file for image {img_path}")
    return image, bboxes, category_ids

def draw_boxes(image, bboxes, color=(255, 0, 0), thickness=2):
    img_copy = image.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

def mosaic_augment(images, boxes_list, labels_list, target_size=TARGET_SIZE):
    """
    Create a mosaic image from 4 images along with their bounding boxes and labels.
    
    Parameters:
        images (list): List of 4 images as NumPy arrays (RGB).
        boxes_list (list): List of lists of boxes (each box: [xmin, ymin, xmax, ymax]) for each image.
        labels_list (list): List of lists of labels corresponding to the boxes.
        target_size (tuple): (width, height) of the output mosaic.
        normalize (bool): If True, convert the mosaic image to float32 and normalize pixel values to [0, 1].
    
    Returns:
        mosaic_img (np.array): The mosaic image.
        mosaic_boxes (list): List of adjusted bounding boxes.
        mosaic_labels (list): List of corresponding labels.
    """
    mosaic_w, mosaic_h = target_size
    # Create a blank canvas (commonly filled with a gray value like 114).
    mosaic_img = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)
    
    # Choose a random center point for the mosaic (to vary placements).
    xc = random.randint(int(mosaic_w * 0.3), int(mosaic_w * 0.7))
    yc = random.randint(int(mosaic_h * 0.3), int(mosaic_h * 0.7))
    
    # Define placements for each of the 4 images:
    # top-left, top-right, bottom-left, bottom-right.
    placements = [
        (0, 0, xc, yc),             # top-left
        (xc, 0, mosaic_w, yc),        # top-right
        (0, yc, xc, mosaic_h),        # bottom-left
        (xc, yc, mosaic_w, mosaic_h)  # bottom-right
    ]
    
    mosaic_boxes = []
    mosaic_labels = []
    
    for i, (img, boxes, labels) in enumerate(zip(images, boxes_list, labels_list)):
        h, w, _ = img.shape
        x1a, y1a, x2a, y2a = placements[i]
        region_w = x2a - x1a
        region_h = y2a - y1a
        
        # Resize the image to fill the assigned quadrant (distortion is acceptable in many YOLO mosaics).
        img_resized = cv2.resize(img, (region_w, region_h))
        mosaic_img[y1a:y2a, x1a:x2a] = img_resized
        
        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            # Compute scaling factors for this image.
            scale_x = region_w / w
            scale_y = region_h / h
            # Scale bounding boxes.
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            # Shift boxes according to the quadrant offset.
            boxes[:, 0] += x1a
            boxes[:, 1] += y1a
            boxes[:, 2] += x1a
            boxes[:, 3] += y1a
            # Clip boxes to mosaic boundaries.
            boxes[:, 0] = np.clip(boxes[:, 0], 0, mosaic_w)
            boxes[:, 1] = np.clip(boxes[:, 1], 0, mosaic_h)
            boxes[:, 2] = np.clip(boxes[:, 2], 0, mosaic_w)
            boxes[:, 3] = np.clip(boxes[:, 3], 0, mosaic_h)
            
            mosaic_boxes.extend(boxes.tolist())
            mosaic_labels.extend(labels)
    
    return mosaic_img, mosaic_boxes, mosaic_labels


############################################### Testing ##################################################################

def test_mosaic():
    # Set dataset folders.
    images_folder = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images/"
    labels_folder = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/labels_minmax/"
    
    # Gather image paths.
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(images_folder, ext)))
    if len(image_paths) < 4:
        raise ValueError("Need at least 4 images for mosaic augmentation test.")
    
    # Randomly select 4 images.
    sample_paths = random.sample(image_paths, 4)
    images = []
    boxes_list = []
    labels_list = []
    
    for path in sample_paths:
        try:
            img, boxes, labels = load_image_and_boxes(path, labels_folder)
            images.append(img)
            boxes_list.append(boxes)
            labels_list.append(labels)
            print(f"Loaded {os.path.basename(path)} with {len(boxes)} boxes.")
        except Exception as e:
            print(e)
    
    # Apply mosaic augmentation.
    mosaic_img, mosaic_boxes, mosaic_labels = mosaic_augment(images, boxes_list, labels_list, target_size=TARGET_SIZE)
    print("\nMosaic Augmentation Test:")
    print(f"Total boxes in mosaic: {len(mosaic_boxes)}")
    print(f"Labels: {mosaic_labels}")
    
    # If the mosaic image is normalized (float in [0,1]), convert to uint8 for drawing.
    disp_img = (mosaic_img * 255).astype(np.uint8) if mosaic_img.dtype == np.float32 else mosaic_img.copy()
    disp_img = disp_img.copy()
    
    # Draw boxes on the mosaic image.
    disp_img = draw_boxes(disp_img, mosaic_boxes, color=(0, 255, 0), thickness=2)
    
    # Display the mosaic.
    plt.figure(figsize=(12, 8))
    plt.imshow(disp_img)
    plt.title("Mosaic Augmented Image with Boxes")
    plt.axis("off")
    plt.show()

def mainTest():
    # Set dataset folders.
    images_folder = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images/"
    labels_folder = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/labels_minmax/"

    # Gather all image paths.
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp')
    all_paths = []
    for ext in exts:
        all_paths.extend(glob.glob(os.path.join(images_folder, ext)))
    if not all_paths:
        print(f"No images found in {images_folder}")
        return

    # Process a random sample (e.g., up to 700 images).
    sample_paths = random.sample(all_paths, min(700, len(all_paths)))

    # Build both pipelines.
    pipeline_yolo = get_train_transforms()
    pipeline_randaug = get_randaugment_pipeline()

    # Initialize statistics.
    stats = {
        "yolo": {"images": 0, "boxes_before": 0, "boxes_after": 0, "boxes_altered": 0},
        "randaug": {"images": 0, "boxes_before": 0, "boxes_after": 0, "boxes_altered": 0},
    }

    # For storing example images along with file name.
    examples_yolo = []
    examples_randaug = []

    for img_path in sample_paths:
        try:
            image, bboxes, cat_ids = load_image_and_boxes(img_path, labels_folder)
        except Exception as e:
            print(e)
            continue

        n_orig = len(bboxes)

        # Process with YOLO-style pipeline.
        data = {"image": image, "bboxes": bboxes, "category_ids": cat_ids}
        augmented_yolo = pipeline_yolo(**data)
        yolo_img = augmented_yolo["image"]
        yolo_bboxes = augmented_yolo["bboxes"]
        stats["yolo"]["images"] += 1
        stats["yolo"]["boxes_before"] += n_orig
        stats["yolo"]["boxes_after"] += len(yolo_bboxes)
        stats["yolo"]["boxes_altered"] += count_altered_boxes(bboxes[:min(n_orig, len(yolo_bboxes))], yolo_bboxes)

        # Process with RandAugment-like pipeline.
        augmented_randaug = pipeline_randaug(**data)
        randaug_img = augmented_randaug["image"]
        randaug_bboxes = augmented_randaug["bboxes"]
        stats["randaug"]["images"] += 1
        stats["randaug"]["boxes_before"] += n_orig
        stats["randaug"]["boxes_after"] += len(randaug_bboxes)
        stats["randaug"]["boxes_altered"] += count_altered_boxes(bboxes[:min(n_orig, len(randaug_bboxes))], randaug_bboxes)

        # Save examples along with the file name.
        examples_yolo.append((img_path, image, yolo_img, bboxes, yolo_bboxes))
        examples_randaug.append((img_path, image, randaug_img, bboxes, randaug_bboxes))

    # Print summary statistics.
    def print_stats(label, d):
        print(f"\n=== Summary for {label} Pipeline ===")
        print(f"Total images processed: {d['images']}")
        print(f"Total bounding boxes before: {d['boxes_before']}")
        print(f"Total bounding boxes after: {d['boxes_after']}")
        print(f"Total bounding boxes altered: {d['boxes_altered']}")
        if d['images'] > 0:
            print(f"Avg boxes per image before: {d['boxes_before']/d['images']:.2f}")
            print(f"Avg boxes per image after: {d['boxes_after']/d['images']:.2f}")
    print_stats("YOLO-style", stats["yolo"])
    print_stats("RandAugment-like", stats["randaug"])

    if examples_yolo:
        file_yolo, orig_img, yolo_img, orig_bboxes, yolo_bboxes = random.choice(examples_yolo)
        print(f"DEBUG: YOLO example from file: {file_yolo}")
        # Resize the original image to TARGET_SIZE.
        orig_resized = cv2.resize(orig_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Compute scaling factors for the original image.
        orig_h, orig_w = orig_img.shape[:2]
        target_w, target_h = TARGET_SIZE
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Adjust the original boxes.
        scaled_orig_bboxes = []
        for bbox in orig_bboxes:
            xmin, ymin, xmax, ymax = bbox[:4]
            scaled_orig_bboxes.append([xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y])
        
        # Combine the original (resized) and augmented images side-by-side.
        combined_yolo = np.hstack([orig_resized, yolo_img])
        
        # Draw the scaled original boxes (red) on the left.
        combined_yolo = draw_boxes(combined_yolo, scaled_orig_bboxes, color=(255, 0, 0))
        
        # Draw augmented boxes (green) on the right, with an x-offset.
        offset = TARGET_SIZE[0]
        for b in yolo_bboxes:
            x1, y1, x2, y2 = b[:4]
            cv2.rectangle(combined_yolo, (int(x1 + offset), int(y1)), (int(x2 + offset), int(y2)), (0, 255, 0), 2)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(combined_yolo, cv2.COLOR_BGR2RGB))
        plt.title(f"YOLO-style: {os.path.basename(file_yolo)}\nOriginal (Left) vs Augmented (Right)")
        plt.axis("off")
        plt.show()

    if examples_randaug:
        file_rand, orig_img, randaug_img, orig_bboxes, randaug_bboxes = random.choice(examples_randaug)
        print(f"DEBUG: RandAugment example from file: {file_rand}")
        # Resize the original image to TARGET_SIZE.
        orig_resized = cv2.resize(orig_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        
        # Compute scaling factors.
        orig_h, orig_w = orig_img.shape[:2]
        target_w, target_h = TARGET_SIZE
        scale_x = target_w / orig_w
        scale_y = target_h / orig_h
        
        # Adjust the original boxes.
        scaled_orig_bboxes = []
        for bbox in orig_bboxes:
            xmin, ymin, xmax, ymax = bbox[:4]
            scaled_orig_bboxes.append([xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y])
        
        # Combine original (resized) and augmented images side-by-side.
        combined_rand = np.hstack([orig_resized, randaug_img])
        
        # Draw scaled original boxes (red) on the left.
        combined_rand = draw_boxes(combined_rand, scaled_orig_bboxes, color=(255, 0, 0))
        
        # Draw augmented boxes (green) on the right, offset by TARGET_SIZE[0].
        offset = TARGET_SIZE[0]
        for b in randaug_bboxes:
            x1, y1, x2, y2 = b[:4]
            cv2.rectangle(combined_rand, (int(x1 + offset), int(y1)), (int(x2 + offset), int(y2)), (0, 255, 0), 2)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(combined_rand, cv2.COLOR_BGR2RGB))
        plt.title(f"RandAugment-like: {os.path.basename(file_rand)}\nOriginal (Left) vs Augmented (Right)")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    #mainTest()
    test_mosaic()
