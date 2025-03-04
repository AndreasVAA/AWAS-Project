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
# TARGET_SIZE is defined as (width, height)
TARGET_SIZE = (1280, 960)

##################################
# DebugWrapper: Prints debug info for selected transforms.
##################################
class DebugWrapper:
    """
    Wraps an Albumentations transform to print debug information
    (image shape and bounding box coordinates) before and after transformation.
    """
    def __init__(self, transform, label=""):
        self.transform = transform
        self.label = label

    def __call__(self, **kwargs):
        image = kwargs.get("image")
        bboxes = kwargs.get("bboxes", [])
        print(f"\nDEBUG [{self.label}] - BEFORE transformation")
        print("  Image shape:", image.shape if image is not None else "None")
        print("  BBoxes before:", bboxes)

        result = self.transform(**kwargs)

        result_image = result.get("image")
        result_bboxes = result.get("bboxes", [])
        print(f"DEBUG [{self.label}] - AFTER transformation")
        print("  Image shape:", result_image.shape if result_image is not None else "None")
        print("  BBoxes after:", result_bboxes)
        return result

    def __getattr__(self, name):
        return getattr(self.transform, name)

##################################
# count_altered_boxes: Compare original vs. augmented boxes.
##################################
def count_altered_boxes(original_boxes, augmented_boxes, tol=1e-3):
    """
    Returns the count of boxes that differ by more than tol for at least one coordinate.
    Compares pairwise for the first min(n_original, n_augmented) boxes.
    """
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
    #    Translate up to 100% in both x and y.
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
    #    Scale from 0.1 to 0.5 (i.e. reduce size) with no translation/rotation/shear.
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
        max_holes=8, max_height=64, max_width=64, p=0.4
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
    # SomeOf block: choose 2 out of a set of transforms with overall probability 0.5.
    # Note: We remove TranslateX/Y (since they don't exist in your Albumentations version)
    some_of = A.SomeOf([
         A.Rotate(limit=30, p=1.0),
         # Simulate translation in x using Affine with only x translation.
         A.Affine(translate_percent={"x": (-0.1, 0.1)}, rotate=0, shear=0, scale=1.0, p=1.0),
         # Simulate translation in y.
         A.Affine(translate_percent={"y": (-0.1, 0.1)}, rotate=0, shear=0, scale=1.0, p=1.0),
         A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=1.0),
         A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
         A.RandomBrightnessContrast(p=1.0)
    ], n=2, replace=False, p=0.5)

    # Wrap the SomeOf block with DebugWrapper.
    transforms_list = [DebugWrapper(some_of, label="RandAugment_SomeOf")]

    # Final resize to TARGET_SIZE.
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
    """
    Blends two images along with their bounding boxes and category ids using mixup augmentation.
    
    If a randomly drawn value exceeds mixup_prob, returns the first image, boxes, and labels unchanged.
    Otherwise, it blends the two images using a ratio drawn from Beta(alpha, alpha).
    The bounding boxes and labels are concatenated.
    
    If the two images have different shapes, image2 is resized to match image1.
    
    Parameters:
      - image1, image2: numpy arrays representing the images.
      - bboxes1, bboxes2: lists of bounding boxes in [xmin, ymin, xmax, ymax] format.
      - cat_ids1, cat_ids2: lists of category IDs corresponding to the bounding boxes.
      - mixup_prob: probability of applying mixup augmentation.
      - alpha: parameter for the Beta distribution.
    
    Returns:
      - mixed_img: the resulting blended image.
      - mixed_bboxes: the concatenated list of bounding boxes.
      - mixed_cat_ids: the concatenated list of category IDs.
    """

    if random.random() > mixup_prob:
        return image1, bboxes1, cat_ids1

    # Ensure both images have the same shape by resizing image2 to match image1 if needed.
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)
    
    ratio = np.random.beta(alpha, alpha)
    mixed_img = (ratio * image1 + (1 - ratio) * image2).astype(image1.dtype)
    mixed_bboxes = bboxes1 + bboxes2
    mixed_cat_ids = cat_ids1 + cat_ids2
    return mixed_img, mixed_bboxes, mixed_cat_ids


##################################
# load_image_and_boxes: Reads an image and its bounding boxes from disk.
##################################
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

##################################
# draw_boxes: Draw bounding boxes on an image.
##################################
def draw_boxes(image, bboxes, color=(255, 0, 0), thickness=2):
    img_copy = image.copy()
    for bbox in bboxes:
        x1, y1, x2, y2 = map(int, bbox[:4])
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, thickness)
    return img_copy

##################################
# Main: Process a sample of images and output summary stats.
##################################
def main():
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

    # Process a random sample (e.g., 5 images).
    sample_paths = random.sample(all_paths, min(700, len(all_paths)))

    # Build both pipelines.
    pipeline_yolo = get_train_transforms()
    pipeline_randaug = get_randaugment_pipeline()

    # Initialize statistics.
    stats = {
        "yolo": {"images": 0, "boxes_before": 0, "boxes_after": 0, "boxes_altered": 0},
        "randaug": {"images": 0, "boxes_before": 0, "boxes_after": 0, "boxes_altered": 0},
    }

    # For storing example images.
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

        # Save one example from each.
        examples_yolo.append((image, yolo_img, bboxes, yolo_bboxes))
        examples_randaug.append((image, randaug_img, bboxes, randaug_bboxes))

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

    # For visualization, choose one random example from each pipeline.
    if examples_yolo:
        ex_yolo = random.choice(examples_yolo)
        orig_img, yolo_img, orig_bboxes, yolo_bboxes = ex_yolo
        # Resize original to TARGET_SIZE for display.
        orig_resized = cv2.resize(orig_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        combined_yolo = np.hstack([orig_resized, yolo_img])
        # Draw original boxes in red on left.
        combined_yolo = draw_boxes(combined_yolo, orig_bboxes, color=(255, 0, 0))
        # Draw augmented boxes in green on right (offset x by TARGET_SIZE[0]).
        offset = TARGET_SIZE[0]
        for b in yolo_bboxes:
            x1, y1, x2, y2 = b[:4]
            cv2.rectangle(combined_yolo, (int(x1 + offset), int(y1)), (int(x2 + offset), int(y2)), (0, 255, 0), 2)
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(combined_yolo, cv2.COLOR_BGR2RGB))
        plt.title("YOLO-style Pipeline: Original (Left) vs Augmented (Right)")
        plt.axis("off")
        plt.show()

    if examples_randaug:
        ex_rand = random.choice(examples_randaug)
        orig_img, randaug_img, orig_bboxes, randaug_bboxes = ex_rand
        orig_resized = cv2.resize(orig_img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
        combined_rand = np.hstack([orig_resized, randaug_img])
        combined_rand = draw_boxes(combined_rand, orig_bboxes, color=(255, 0, 0))
        offset = TARGET_SIZE[0]
        for b in randaug_bboxes:
            x1, y1, x2, y2 = b[:4]
            cv2.rectangle(combined_rand, (int(x1 + offset), int(y1)), (int(x2 + offset), int(y2)), (0, 255, 0), 2)
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(combined_rand, cv2.COLOR_BGR2RGB))
        plt.title("RandAugment-like Pipeline: Original (Left) vs Augmented (Right)")
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    main()
