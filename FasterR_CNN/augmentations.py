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
# MAIN AUGMENTATION FUNCTIONS
##################################

def get_train_transforms(
    fliplr=0.5,
    erasing=0.4,
    auto_augment=True,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    translate=0.1,
    scale=0.5,
    rotate=0.0,
    shear=0.0,
    crop_fraction=1.0
):
    """
    Returns an Albumentations Compose pipeline that applies YOLO-style augmentations.
    The final output image is resized to TARGET_SIZE (width x height).
    BBoxes are expected in 'pascal_voc' format: [xmin, ymin, xmax, ymax].
    """
    transforms_list = []

    # 1. Horizontal flip
    transforms_list.append(A.HorizontalFlip(p=fliplr))

    # 2. HSV augmentation
    hue_shift = int(hsv_h * 180)
    sat_shift = int(hsv_s * 255)
    val_shift = int(hsv_v * 255)
    transforms_list.append(A.HueSaturationValue(
        hue_shift_limit=hue_shift,
        sat_shift_limit=sat_shift,
        val_shift_limit=val_shift,
        p=0.5
    ))

    # 3. Affine transformation
    if shear == 0:
        transforms_list.append(A.ShiftScaleRotate(
            shift_limit=translate,
            scale_limit=scale,
            rotate_limit=(-rotate, rotate),
            border_mode=cv2.BORDER_CONSTANT,
            value=(114,114,114),
            p=0.5
        ))
    else:
        # If you want shear, you could use A.Affine but remove 'mode' and 'cval'
        transforms_list.append(A.Affine(
            translate_percent={"x": translate, "y": translate},
            scale_limit=scale,
            rotate_limit=(-rotate, rotate),
            shear=(-shear, shear),
            p=0.5
        ))

    # 4. Auto augmentation using OneOf fallback (since RandAugment not available).
    if auto_augment:
        transforms_list.append(A.OneOf([
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
            A.RGBShift(p=0.5)
        ], p=0.5))

    # 5. Random erasing (CoarseDropout)
    transforms_list.append(A.CoarseDropout(
        max_holes=8, max_height=16, max_width=16,
        min_holes=1, fill_value=114, p=erasing
    ))

    # 6. Optional random crop if crop_fraction < 1.0
    if crop_fraction < 1.0:
        w = int(TARGET_SIZE[0] * crop_fraction)
        h = int(TARGET_SIZE[1] * crop_fraction)
        transforms_list.append(A.RandomCrop(width=w, height=h, p=0.5))

    # 7. Final resize to TARGET_SIZE
    transforms_list.append(A.Resize(
        width=TARGET_SIZE[0],
        height=TARGET_SIZE[1],
        interpolation=cv2.INTER_LINEAR,
        p=1.0
    ))

    return A.Compose(
        transforms_list,
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'])
    )


def mosaic_augment(
    images,
    bboxes_list,
    category_ids_list,
    mosaic_prob=0.0, #### Changing to 0 to turn of
    mosaic_size=TARGET_SIZE,
    min_area_visibility=0.8 ###### Change this for affecting bounding boxes visibility
):
    """
    Mosaic augmentation with an additional check:
    We only keep a bounding box if the clipped area is at least
    `min_area_visibility` fraction (e.g. 0.8) of the original area.
    The final mosaic is resized to mosaic_size.
    """
    if random.random() > mosaic_prob or len(images) < 4:
        return images[0], bboxes_list[0], category_ids_list[0]

    target_w, target_h = mosaic_size
    canvas = np.full((target_h * 2, target_w * 2, 3), 114, dtype=images[0].dtype)

    # Random mosaic center
    xc = int(random.uniform(target_w * 0.5, target_w * 1.5))
    yc = int(random.uniform(target_h * 0.5, target_h * 1.5))

    mosaic_bboxes = []
    mosaic_cat_ids = []

    for i, (img, bboxes, cat_ids) in enumerate(zip(images, bboxes_list, category_ids_list)):
        h, w = img.shape[:2]

        if i == 0:  # top-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
        elif i == 1:  # top-right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_w * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - xc), h
        elif i == 2:  # bottom-left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_h * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - yc, h)
        else:  # bottom-right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_w * 2), min(yc + h, target_h * 2)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - xc), min(h, y2a - yc)

        canvas[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]

        pad_x = x1a - x1b
        pad_y = y1a - y1b
        if len(bboxes) > 0:
            bboxes = np.array(bboxes)
            for idx, box in enumerate(bboxes):
                orig_xmin, orig_ymin, orig_xmax, orig_ymax = box
                orig_w = (orig_xmax - orig_xmin)
                orig_h = (orig_ymax - orig_ymin)
                orig_area = orig_w * orig_h

                # SHIFT
                shifted_xmin = orig_xmin + pad_x
                shifted_ymin = orig_ymin + pad_y
                shifted_xmax = orig_xmax + pad_x
                shifted_ymax = orig_ymax + pad_y

                # CLIP
                clipped_xmin = np.clip(shifted_xmin, 0, target_w * 2)
                clipped_ymin = np.clip(shifted_ymin, 0, target_h * 2)
                clipped_xmax = np.clip(shifted_xmax, 0, target_w * 2)
                clipped_ymax = np.clip(shifted_ymax, 0, target_h * 2)

                clipped_w = clipped_xmax - clipped_xmin
                clipped_h = clipped_ymax - clipped_ymin
                if clipped_w <= 0 or clipped_h <= 0:
                    continue  # fully out of mosaic region

                clipped_area = clipped_w * clipped_h
                area_ratio = clipped_area / (orig_area + 1e-6)

                # Keep only if area_ratio >= min_area_visibility
                if area_ratio >= min_area_visibility:
                    mosaic_bboxes.append([
                        clipped_xmin, clipped_ymin,
                        clipped_xmax, clipped_ymax
                    ])
                    mosaic_cat_ids.append(cat_ids[idx])

    # Resize final mosaic to target size
    mosaic_img = cv2.resize(canvas, (target_w, target_h))
    return mosaic_img, mosaic_bboxes, mosaic_cat_ids


def mixup_augment(
    image1, bboxes1, cat_ids1,
    image2, bboxes2, cat_ids2,
    mixup_prob=0.0, alpha=32
):
    if random.random() > mixup_prob:
        return image1, bboxes1, cat_ids1

    ratio = np.random.beta(alpha, alpha)
    mixed_img = (ratio * image1 + (1 - ratio) * image2).astype(image1.dtype)
    mixed_bboxes = bboxes1 + bboxes2
    mixed_cat_ids = cat_ids1 + cat_ids2
    return mixed_img, mixed_bboxes, mixed_cat_ids

##################################
# TEST BLOCK
##################################
if __name__ == "__main__":

    # First image + annotation
    test_img_path = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images/gunnerusvertikal2_14.jpg"
    annotation_path = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/labels_minmax/gunnerusvertikal2_14.txt"

    # Second image + annotation for Mixup
    second_img_path = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images/zooplanktonwp2_44.jpg"
    second_annotation_path = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/labels_minmax/zooplanktonwp2_44.txt"

    # LOAD FIRST IMAGE
    orig_img_bgr = cv2.imread(test_img_path)
    if orig_img_bgr is None:
        print("Test image not found at", test_img_path)
        exit()
    orig_img = cv2.cvtColor(orig_img_bgr, cv2.COLOR_BGR2RGB)

    # LOAD FIRST ANNOTATION
    sample_bboxes = []
    sample_cat_ids = []
    with open(annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x_min = float(parts[1])
                y_min = float(parts[2])
                x_max = float(parts[3])
                y_max = float(parts[4])
                sample_bboxes.append([x_min, y_min, x_max, y_max])
                sample_cat_ids.append(cls_id)

    # LOAD SECOND IMAGE (for Mixup)
    second_img_bgr = cv2.imread(second_img_path)
    if second_img_bgr is None:
        print("Second image not found at", second_img_path)
        exit()
    second_img = cv2.cvtColor(second_img_bgr, cv2.COLOR_BGR2RGB)

    # LOAD SECOND ANNOTATION
    second_bboxes = []
    second_cat_ids = []
    with open(second_annotation_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls_id = int(parts[0])
                x_min = float(parts[1])
                y_min = float(parts[2])
                x_max = float(parts[3])
                y_max = float(parts[4])
                second_bboxes.append([x_min, y_min, x_max, y_max])
                second_cat_ids.append(cls_id)

    print("FIRST IMAGE BOXES:", sample_bboxes)
    print("SECOND IMAGE BOXES:", second_bboxes)

    # ============ Test 1: Albumentations pipeline =============
    transform = get_train_transforms()
    alb_input = {
        "image": orig_img,
        "bboxes": sample_bboxes,
        "category_ids": sample_cat_ids
    }
    augmented = transform(**alb_input)
    aug_img = augmented['image']
    aug_bboxes = augmented['bboxes']

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(orig_img)
    for bbox in sample_bboxes:
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2]-bbox[0],
                             bbox[3]-bbox[1],
                             fill=False, edgecolor='red', linewidth=2)
        ax[0].add_patch(rect)
    ax[0].set_title("Original (1) with Annotations")

    ax[1].imshow(aug_img)
    for bbox in aug_bboxes:
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2]-bbox[0],
                             bbox[3]-bbox[1],
                             fill=False, edgecolor='red', linewidth=2)
        ax[1].add_patch(rect)
    ax[1].set_title("Albumentations Augmentation")
    plt.show()

    # ============ Test 2: Mosaic with IoU-based filtering ============
    images = [orig_img.copy() for _ in range(4)]
    bboxes_list = [sample_bboxes for _ in range(4)]
    cat_ids_list = [sample_cat_ids for _ in range(4)]
    mosaic_img, mosaic_bboxes, mosaic_cat_ids = mosaic_augment(
        images, bboxes_list, cat_ids_list,
        mosaic_prob=1.0,
        mosaic_size=TARGET_SIZE,
        min_area_visibility=0.8  # Keep only boxes with >=80% area retained
    )

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(orig_img)
    for bbox in sample_bboxes:
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2]-bbox[0],
                             bbox[3]-bbox[1],
                             fill=False, edgecolor='red', linewidth=2)
        ax[0].add_patch(rect)
    ax[0].set_title("Original (1) with Annotations")

    ax[1].imshow(mosaic_img)
    for bbox in mosaic_bboxes:
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2]-bbox[0],
                             bbox[3]-bbox[1],
                             fill=False, edgecolor='red', linewidth=2)
        ax[1].add_patch(rect)
    ax[1].set_title("Mosaic w/ 80% Visibility Filter")
    plt.show()

    # ============ Test 3: Mixup with a SECOND IMAGE ============
    # Blend the first image with the second one so you see an actual difference.
    mixup_img, mixup_bboxes, mixup_cat_ids = mixup_augment(
        orig_img, sample_bboxes, sample_cat_ids,
        second_img, second_bboxes, second_cat_ids,
        mixup_prob=1.0,  # Force mixup
        alpha=32
    )
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Left: show first image with boxes
    ax[0].imshow(orig_img)
    for bbox in sample_bboxes:
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2]-bbox[0],
                             bbox[3]-bbox[1],
                             fill=False, edgecolor='red', linewidth=2)
        ax[0].add_patch(rect)
    ax[0].set_title("Original (1) with Annotations")

    # Right: blended image
    ax[1].imshow(mixup_img)
    for bbox in mixup_bboxes:
        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2]-bbox[0],
                             bbox[3]-bbox[1],
                             fill=False, edgecolor='red', linewidth=2)
        ax[1].add_patch(rect)
    ax[1].set_title("Mixup (1 & 2) with BBoxes Combined")
    plt.show()
