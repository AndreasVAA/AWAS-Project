import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import logging

# Import your dataset and model functions
from Faster_R_CNN import AbsoluteDataset, get_model, collate_fn

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AVAenv")

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    union_area = (box1[2]-box1[0])*(box1[3]-box1[1]) + (box2[2]-box2[0])*(box2[3]-box2[1]) - inter_area
    return inter_area / union_area

def convert_gt_to_coco(dataset):
    coco_gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "plankton", "supercategory": "none"}]
    }
    ann_id = 1
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        image_id = int(target["image_id"].item())
        height, width = img.shape[1], img.shape[2]
        coco_gt["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"image_{image_id}.jpg"  # Placeholder filename.
        })
        boxes = target["boxes"].numpy()
        labels = target["labels"].numpy()
        iscrowd = target["iscrowd"].numpy()
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            w = xmax - xmin
            h = ymax - ymin
            coco_gt["annotations"].append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(xmin), float(ymin), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": int(iscrowd[i])
            })
            ann_id += 1
    return coco_gt

def convert_preds_to_coco(all_preds):
    coco_results = []
    for pred in all_preds:
        image_id = int(pred["image_id"].item())
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        labels = pred["labels"].cpu().numpy()
        for i in range(len(boxes)):
            xmin, ymin, xmax, ymax = boxes[i]
            w = xmax - xmin
            h = ymax - ymin
            coco_results.append({
                "image_id": image_id,
                "category_id": int(labels[i]),
                "bbox": [float(xmin), float(ymin), float(w), float(h)],
                "score": float(scores[i])
            })
    return coco_results

def coco_evaluation(gt_json_path, pred_json_path):
    coco_gt = COCO(gt_json_path)
    coco_dt = coco_gt.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

if __name__ == '__main__':
    # Update these paths as needed.
    val_images_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/images"
    val_labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels_minmax"
    model_path = "/home/itk/Desktop/Andreas/AWAS-Project/FasterR_CNN/best_model_absolute.pth"

    score_threshold = 0.3  # Minimum confidence for predictions
    iou_threshold = 0.3    # (Unused here, kept for potential future use)

    logger.info("Initializing validation dataset...")
    val_dataset = AbsoluteDataset(val_images_dir, val_labels_dir)
    # Use a subset (e.g., 50 images) for quick evaluation.
    val_subset = Subset(val_dataset, list(range(min(50, len(val_dataset)))))
    batch_size = 2
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    logger.info(f"Loaded {len(val_subset)} validation samples.")

    # Setup and load the model.
    num_classes = 2  # For example: 1 class ("plankton") + background.
    model = get_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    logger.info(f"Loading model weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    # If your checkpoint is a dict with extra keys, extract the model state dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    # Run inference on the validation set.
    all_preds = []
    for images, targets in val_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        # Append predictions with the image id from the corresponding target.
        for output, target in zip(outputs, targets):
            output["image_id"] = target["image_id"]
            all_preds.append(output)

    # Convert ground truth and predictions to COCO format.
    logger.info("Converting ground truth annotations to COCO format...")
    coco_gt = convert_gt_to_coco(val_dataset)
    gt_json_path = "gt_annotations.json"
    with open(gt_json_path, "w") as f:
        json.dump(coco_gt, f)

    logger.info("Converting model predictions to COCO format...")
    coco_results = convert_preds_to_coco(all_preds)
    pred_json_path = "pred_annotations.json"
    with open(pred_json_path, "w") as f:
        json.dump(coco_results, f)

    # Run COCO evaluation and display the results.
    logger.info("Running COCO evaluation...")
    stats = coco_evaluation(gt_json_path, pred_json_path)
    logger.info("COCO Evaluation Stats:")
    logger.info(stats)
