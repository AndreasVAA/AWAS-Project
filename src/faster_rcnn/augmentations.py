#!/usr/bin/env python3
import os
import glob
import cv2
import numpy as np
import albumentations as A
import random
import matplotlib.pyplot as plt

##################################
# DebugWrapper: Wraps an Albumentations transform.
##################################




class DebugWrapper:
    def __init__(self, transform, label=""):
        self.transform = transform
        self.label = label

    def __call__(self, **kwargs):
        return self.transform(**kwargs)

    def __getattr__(self, name):
        return getattr(self.transform, name)

##################################
# YOLO-style Pipeline
##################################
def get_train_transforms(target_size):
    transforms_list = [
        A.HorizontalFlip(p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=30, p=0.5),
        DebugWrapper(A.Affine(translate_percent={"x": (0.0, 1.0), "y": (0.0, 1.0)}, rotate=0, shear=0, scale=1.0, p=0.1), label="Affine_TranslateOnly"),
        DebugWrapper(A.Affine(translate_percent=0, rotate=0, shear=0, scale=(0.1, 0.5), p=0.5), label="Affine_ScaleOnly"),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(64, 64), hole_width_range=(64, 64), fill_value=0, p=0.4),
        A.Resize(width=target_size[0], height=target_size[1], interpolation=cv2.INTER_LINEAR, p=1.0)
    ]
    return A.Compose(transforms_list, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.2))

##################################
# RandAugment-like Pipeline
##################################
def get_randaugment_pipeline(target_size):
    some_of = A.SomeOf([
        A.Rotate(limit=30, p=1.0),
        A.Affine(translate_percent={"x": (-0.1, 0.1)}, rotate=0, shear=0, scale=1.0, p=1.0),
        A.Affine(translate_percent={"y": (-0.1, 0.1)}, rotate=0, shear=0, scale=1.0, p=1.0),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=30, p=1.0),
        A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
        A.RandomBrightnessContrast(p=1.0)
    ], n=2, replace=False, p=0.5)
    transforms_list = [
        DebugWrapper(some_of, label="RandAugment_SomeOf"),
        A.Resize(width=target_size[0], height=target_size[1], interpolation=cv2.INTER_LINEAR, p=1.0)
    ]
    return A.Compose(transforms_list, bbox_params=A.BboxParams(format='pascal_voc', label_fields=['category_ids'], min_visibility=0.2))

##################################
# Mixup Augmentation
##################################
def mixup_augment(image1, bboxes1, cat_ids1, image2, bboxes2, cat_ids2, mixup_prob=0.0, alpha=32):
    if random.random() > mixup_prob:
        return image1, bboxes1, cat_ids1
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]), interpolation=cv2.INTER_LINEAR)
    ratio = np.random.beta(alpha, alpha)
    mixed_img = (ratio * image1 + (1 - ratio) * image2).astype(image1.dtype)
    return mixed_img, bboxes1 + bboxes2, cat_ids1 + cat_ids2

##################################
# Mosaic Augmentation
##################################
def mosaic_augment(images, boxes_list, labels_list, target_size):
    mosaic_w, mosaic_h = target_size
    mosaic_img = np.full((mosaic_h, mosaic_w, 3), 114, dtype=np.uint8)
    xc = random.randint(int(mosaic_w * 0.3), int(mosaic_w * 0.7))
    yc = random.randint(int(mosaic_h * 0.3), int(mosaic_h * 0.7))
    placements = [
        (0, 0, xc, yc),
        (xc, 0, mosaic_w, yc),
        (0, yc, xc, mosaic_h),
        (xc, yc, mosaic_w, mosaic_h)
    ]
    mosaic_boxes = []
    mosaic_labels = []
    for i, (img, boxes, labels) in enumerate(zip(images, boxes_list, labels_list)):
        h, w, _ = img.shape
        x1a, y1a, x2a, y2a = placements[i]
        region_w = x2a - x1a
        region_h = y2a - y1a
        img_resized = cv2.resize(img, (region_w, region_h))
        mosaic_img[y1a:y2a, x1a:x2a] = img_resized
        if boxes:
            boxes = np.array(boxes, dtype=np.float32)
            scale_x = region_w / w
            scale_y = region_h / h
            boxes[:, [0, 2]] *= scale_x
            boxes[:, [1, 3]] *= scale_y
            boxes[:, 0] += x1a
            boxes[:, 1] += y1a
            boxes[:, 2] += x1a
            boxes[:, 3] += y1a
            boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, mosaic_w)
            boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, mosaic_h)
            mosaic_boxes.extend(boxes.tolist())
            mosaic_labels.extend(labels)
    return mosaic_img, mosaic_boxes, mosaic_labels