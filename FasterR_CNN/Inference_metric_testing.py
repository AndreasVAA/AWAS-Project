import os
import cv2
import json
import time
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix, classification_report
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

#############################################
# Dataset and Model Definitions
#############################################

from Faster_R_CNN import AbsoluteDataset, get_model, collate_fn


#############################################
# Metric Functions
#############################################

def compute_iou(box1, box2):
    """
    Computes the Intersection over Union (IoU) between two bounding boxes.
    Each box is in the format [xmin, ymin, xmax, ymax].
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

def convert_gt_to_coco(dataset):
    """
    Converts the ground truth annotations from a dataset into COCO format.
    Assumes that __getitem__ returns (image, target) where target has keys:
    boxes, labels, image_id, area, iscrowd.
    """
    coco_gt = {"images": [], "annotations": [], "categories": []}
    # For a single class ("plankton") assume id=1.
    coco_gt["categories"].append({"id": 1, "name": "plankton", "supercategory": "none"})
    ann_id = 1
    for idx in range(len(dataset)):
        img, target = dataset[idx]
        image_id = int(target["image_id"].item())
        height, width = img.shape[1], img.shape[2]
        coco_gt["images"].append({
            "id": image_id,
            "width": width,
            "height": height,
            "file_name": f"image_{image_id}.jpg"  # Placeholder name.
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
    print(f"[COCO] Converted {len(coco_gt['annotations'])} annotations from {len(coco_gt['images'])} images.")
    return coco_gt

def convert_preds_to_coco(all_preds):
    """
    Converts a list of predictions into COCO result format.
    Each prediction should be a dictionary with keys: boxes, scores, labels, image_id.
    """
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
    print(f"[COCO] Converted {len(coco_results)} prediction results.")
    return coco_results

def coco_evaluation(gt_json_path, pred_json_path):
    """
    Runs COCO evaluation given ground truth and prediction JSON files.
    Returns the COCO evaluation statistics.
    """
    print(f"[COCO Eval] Loading GT from {gt_json_path} and predictions from {pred_json_path}.")
    coco_gt_api = COCO(gt_json_path)
    coco_dt_api = coco_gt_api.loadRes(pred_json_path)
    coco_eval = COCOeval(coco_gt_api, coco_dt_api, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats

#############################################
# Evaluation Script
#############################################
if __name__ == '__main__':
    #########################################
    # Settings and Paths
    #########################################
    val_images_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/images"       # Update with your validation images directory.
    val_labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels_minmax"  # Update with your absolute label folder.
    model_path = "/home/itk/Desktop/Andreas/AWAS-Project/FasterR_CNN/best_model_absolute.pth"         # Model weights saved after training.

    score_threshold = 0.3  # Minimum confidence to consider a prediction.
    iou_threshold = 0.3    # IoU threshold for matching.
    
    #########################################
    # Initialize Validation Dataset and DataLoader
    #########################################
    print("[Eval] Initializing validation dataset...")
    val_dataset = AbsoluteDataset(val_images_dir, val_labels_dir)
    # Use a subset for quicker evaluation (e.g., 50 images).
    val_subset = Subset(val_dataset, list(range(min(50, len(val_dataset)))))
    batch_size = 4
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"[Eval] Loaded {len(val_subset)} validation samples.")

    #########################################
    # Setup and Load the Model
    #########################################
    num_classes = 2  # e.g., 1 class ("plankton") + background.
    model = get_model(num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"[Eval] Loading model weights from {model_path}...")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    #########################################
    # Run Inference and Compute Instance-Level Metrics
    #########################################
    total_tp = 0
    total_fp = 0
    total_fn = 0
    all_preds = []  # For COCO evaluation.
    
    print("[Eval] Running inference on the validation set...")
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
                gt_boxes = target["boxes"].cpu().numpy()  # Ground truth boxes.
                pred_boxes = output["boxes"].cpu().numpy()  # Predicted boxes.
                pred_scores = output["scores"].cpu().numpy()
                
                # Consider only predictions above the confidence threshold.
                valid_idx = np.where(pred_scores >= score_threshold)[0]
                pred_boxes = pred_boxes[valid_idx]
                
                matched = set()
                tp = 0
                for pred_box in pred_boxes:
                    for i, gt_box in enumerate(gt_boxes):
                        if i in matched:
                            continue
                        if compute_iou(pred_box, gt_box) >= iou_threshold:
                            tp += 1
                            matched.add(i)
                            break
                fp = len(pred_boxes) - tp
                fn = len(gt_boxes) - tp

                total_tp += tp
                total_fp += fp
                total_fn += fn
                
                # *** Fix: Add the image_id from the target to the output ***
                output["image_id"] = target["image_id"]
                all_preds.append(output)

    
    precision_inst = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall_inst = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1_inst = 2 * precision_inst * recall_inst / (precision_inst + recall_inst) if (precision_inst + recall_inst) > 0 else 0.0
    
    print("\n[Instance-Level Evaluation] Counts:")
    print(f"  True Positives (TP):  {total_tp}")
    print(f"  False Positives (FP): {total_fp}")
    print(f"  False Negatives (FN): {total_fn}")
    print(f"  Precision: {precision_inst:.4f}, Recall: {recall_inst:.4f}, F1 Score: {f1_inst:.4f}")
    
    #########################################
    # scikit-learn Evaluation (Instance-Level)
    #########################################
    # We construct synthetic arrays:
    #   For each ground truth object (TP + FN), label as 1.
    #   For each false positive, we add a predicted positive without a corresponding ground truth.
    y_true = [1] * (total_tp + total_fn) + [0] * (total_fp)
    y_pred = [1] * total_tp + [0] * total_fn + [1] * (total_fp)
    
    print("\n[sklearn Evaluation] Computing confusion matrix and classification report...")
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, digits=4)
    print("Confusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)
    
    #########################################
    # COCO Evaluation
    #########################################
    print("\n[COCO Evaluation] Converting ground truth annotations to COCO format...")
    coco_gt = convert_gt_to_coco(val_dataset)
    gt_json_path = "gt_annotations.json"
    with open(gt_json_path, "w") as f:
        json.dump(coco_gt, f)
    print(f"[COCO Evaluation] Ground truth saved to {gt_json_path}.")
    
    print("\n[COCO Evaluation] Converting model predictions to COCO format...")
    coco_results = convert_preds_to_coco(all_preds)
    if len(coco_results) == 0:
        print("[COCO Evaluation] No predictions were generated. Skipping COCO evaluation.")
    else:
        pred_json_path = "pred_annotations.json"
        with open(pred_json_path, "w") as f:
            json.dump(coco_results, f)
        print(f"[COCO Evaluation] Predictions saved to {pred_json_path}.")
        
        print("[COCO Evaluation] Running COCO evaluation...")
        coco_stats = coco_evaluation(gt_json_path, pred_json_path)
        print("\nCOCO Evaluation Stats:")
        print(coco_stats)

    
    #########################################
    # End of Evaluation Script
    #########################################
