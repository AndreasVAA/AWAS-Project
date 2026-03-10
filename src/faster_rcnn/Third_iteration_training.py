import os
import cv2
import torch
import numpy as np
import random
import json
import time
import logging
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.transforms.functional as TF
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import sys

# === BASE CONFIGURATION (Defaults that can be overridden by TRAINING_VARIATIONS) ===
BASE_CONFIG = {
    "NUM_EPOCHS": 400,
    "COCO_EVAL_MIN_SCORE_THRESHOLD": 0.01,
    "EARLY_STOPPING_PATIENCE": 40,
    "NUM_WORKERS": 4,
    "SEED": 42,
    "SCHEDULER_PATIENCE": 10,
    "DEBUG_LOGGING": False,
    "WARMUP_EPOCHS": 3,
    "OUTPUT_BASE_DIR": "FasterRCNN_Runs_Third_iteraion", # NEW: Configurable base output directory
    # Default SCORE_THRESHOLD and IOU_THRESHOLD for F1-based optimization (can be overridden in variations)
    "SCORE_THRESHOLD": 0.4, 
    "IOU_THRESHOLD": 0.6,
}

# === PATHS & LISTS FOR SPECIFIC DATASETS ===
SINGLE_CLASS_BASE_PATH = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318"
SINGLE_CLASS_NAME_LIST = ["Plankton"] # Only one class

MULTI_CLASS_TRAIN_IMG_DIR = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/images"
MULTI_CLASS_TRAIN_LBL_DIR = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/labels_minmax"
MULTI_CLASS_VAL_IMG_DIR =   "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/val/images"
MULTI_CLASS_VAL_LBL_DIR =   "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/val/labels_minmax"
MULTI_CLASS_NAME_LIST = ["Tripos longipes", "Tripos fusus", "Tripos lineatum/furca", "Chaetoceros", "Coscinodiscus"]


# === NEW TRAINING VARIATIONS (F1-Optimized, Single-Class Focus) ===
TRAINING_VARIATIONS = [
    {
        "VARIATION_NAME": "SGD_MultiClass_640_NoAccum_F1Opt",
        "CLASS_NAMES": MULTI_CLASS_NAME_LIST,
        "TRAIN_IMAGE_DIR": MULTI_CLASS_TRAIN_IMG_DIR,
        "TRAIN_LABEL_DIR": MULTI_CLASS_TRAIN_LBL_DIR,
        "VAL_IMAGE_DIR":   MULTI_CLASS_VAL_IMG_DIR,
        "VAL_LABEL_DIR":   MULTI_CLASS_VAL_LBL_DIR,
        "TARGET_SIZE": (640, 640),      # (width, height)
        "BATCH_SIZE": 4,
        "OPTIMIZER_TYPE": "SGD",
        "LEARNING_RATE": 0.005,
        "SGD_MOMENTUM": 0.937,
        "WEIGHT_DECAY": 0.0005,
        "USE_GRADIENT_ACCUMULATION": False,
        "SCORE_THRESHOLD": 0.4,
        "IOU_THRESHOLD": 0.6,
    },
    {
        "VARIATION_NAME": "SGD_MultiClass_1280_NoAccum_F1Opt",
        "CLASS_NAMES": MULTI_CLASS_NAME_LIST,
        "TRAIN_IMAGE_DIR": MULTI_CLASS_TRAIN_IMG_DIR,
        "TRAIN_LABEL_DIR": MULTI_CLASS_TRAIN_LBL_DIR,
        "VAL_IMAGE_DIR":   MULTI_CLASS_VAL_IMG_DIR,
        "VAL_LABEL_DIR":   MULTI_CLASS_VAL_LBL_DIR,
        "TARGET_SIZE": (1280, 1280),    # (width, height)
        "BATCH_SIZE": 4,
        "OPTIMIZER_TYPE": "SGD",
        "LEARNING_RATE": 0.005,
        "SGD_MOMENTUM": 0.937,
        "WEIGHT_DECAY": 0.0005,
        "USE_GRADIENT_ACCUMULATION": False,
        "SCORE_THRESHOLD": 0.4,
        "IOU_THRESHOLD": 0.6,
    },
    
    # { # Example of how you might re-enable a multi-class run later
    #     "VARIATION_NAME": "SGD_MultiClass_1280_NoAccum_F1Opt",
    #     "CLASS_NAMES": MULTI_CLASS_NAME_LIST,
    #     "TRAIN_IMAGE_DIR": MULTI_CLASS_TRAIN_IMG_DIR,
    #     "TRAIN_LABEL_DIR": MULTI_CLASS_TRAIN_LBL_DIR,
    #     "VAL_IMAGE_DIR":   MULTI_CLASS_VAL_IMG_DIR,
    #     "VAL_LABEL_DIR":   MULTI_CLASS_VAL_LBL_DIR,
    #     "TARGET_SIZE": (1280, 1280),
    #     "BATCH_SIZE": 4,
    #     "OPTIMIZER_TYPE": "SGD",
    #     "LEARNING_RATE": 0.005,
    #     "SGD_MOMENTUM": 0.937,
    #     "WEIGHT_DECAY": 0.0005,
    #     "USE_GRADIENT_ACCUMULATION": False,
    #     "SCORE_THRESHOLD": 0.4, # For F1 calculation during training validation
    #     "IOU_THRESHOLD": 0.6,   # For F1 calculation during training validation
    # },
]

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0]); y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2]); y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1); inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

class AbsoluteDataset(Dataset): # Handles resizing and original absolute pixel coordinates
    def __init__(self, images_dir, labels_dir, target_image_size, class_names_for_this_dataset):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_width, self.target_height = target_image_size # e.g., (640, 640) or (1280, 1280)
        self.class_names = class_names_for_this_dataset
        self.images = []

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        for f_name in sorted(os.listdir(images_dir)):
            if f_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                label_path = os.path.join(self.labels_dir, os.path.splitext(f_name)[0] + ".txt")
                if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
                    self.images.append(f_name)
                elif not os.path.exists(label_path):
                     logging.debug(f"No label file for image '{f_name}' in {self.labels_dir}. Skipping image.")
                else:
                     logging.debug(f"Empty label file for image '{f_name}' in {self.labels_dir}. Skipping image.")
        
        if not self.images:
            logging.warning(f"No valid images with non-empty labels found in {images_dir} (labels in {labels_dir}).")
        else:
            logging.info(f"AbsoluteDataset: Initialized with {len(self.images)} images from {images_dir}.")


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.images_dir, img_file)
        img = cv2.imread(img_path)
        
        if img is None:
            logging.error(f"Failed to load image: {img_path}. Using black placeholder matching target size.")
            img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8) # Placeholder is RGB
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        original_height, original_width = img.shape[:2]
        
        # Resize image to target size
        img_resized = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

        label_path = os.path.join(self.labels_dir, os.path.splitext(img_file)[0] + '.txt')
        boxes, labels_for_target = [], [] # Renamed to avoid conflict with 'labels' module

        if os.path.exists(label_path):
            with open(label_path) as f:
                for line_num, line in enumerate(f):
                    parts = line.strip().split()
                    if not parts: continue
                    try:
                        float_parts = [float(p) for p in parts]
                    except ValueError:
                        logging.warning(f"L{line_num+1} in {label_path}: Non-float value '{line.strip()}'. Skipping.")
                        continue
                    if len(float_parts) != 5: # class_id, xmin, ymin, xmax, ymax
                        logging.warning(f"L{line_num+1} in {label_path}: Expected 5 parts for absolute pixel format, got {len(float_parts)} ('{line.strip()}'). Skipping.")
                        continue

                    file_cls_id_from_file = int(float_parts[0])
                    if not (1 <= file_cls_id_from_file <= len(self.class_names)):
                        logging.warning(f"Class ID {file_cls_id_from_file} from file {label_path} (line {line_num+1}) "
                                        f"is out of expected range [1, {len(self.class_names)}] for "
                                        f"defined classes: {self.class_names}. Skipping annotation.")
                        continue
                    model_label = file_cls_id_from_file

                    # These are absolute pixel coordinates for the *original* image
                    xmin_orig, ymin_orig, xmax_orig, ymax_orig = float_parts[1], float_parts[2], float_parts[3], float_parts[4]
                    
                    # Calculate scaling factors
                    if original_width == 0 or original_height == 0: # Should not happen if image loaded
                        logging.error(f"Image {img_file} has zero original dimension. Skipping annotation scaling.")
                        continue
                        
                    x_scale = self.target_width / original_width
                    y_scale = self.target_height / original_height

                    # Scale absolute coordinates to the resized image dimensions
                    x1_s = xmin_orig * x_scale
                    y1_s = ymin_orig * y_scale
                    x2_s = xmax_orig * x_scale
                    y2_s = ymax_orig * y_scale
                    
                    # Clamp to resized image dimensions
                    x1_s = max(0.0, x1_s)
                    y1_s = max(0.0, y1_s)
                    x2_s = min(float(self.target_width), x2_s)  # Clamp to target_width
                    y2_s = min(float(self.target_height), y2_s) # Clamp to target_height

                    if x2_s > x1_s and y2_s > y1_s: # Ensure valid box after scaling and clamping
                        boxes.append([x1_s, y1_s, x2_s, y2_s])
                        labels_for_target.append(model_label)
        
        img_tensor = TF.to_tensor(img_resized.copy()) # Operate on the resized image
        
        target = {"image_id": torch.tensor([idx], dtype=torch.int64)}
        if boxes:
            target_boxes = torch.tensor(boxes, dtype=torch.float32)
            target["boxes"] = target_boxes
            target["labels"] = torch.tensor(labels_for_target, dtype=torch.int64) # Model expects 1-indexed labels
            target["area"] = (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1])
            target["iscrowd"] = torch.zeros((len(boxes),), dtype=torch.int64)
        else: 
            target["boxes"] = torch.empty((0, 4), dtype=torch.float32)
            target["labels"] = torch.empty((0,), dtype=torch.int64)
            target["area"] = torch.empty((0,), dtype=torch.float32)
            target["iscrowd"] = torch.empty((0,), dtype=torch.int64)
            
        return img_tensor, target

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

def get_model(num_model_classes): # num_model_classes = actual classes + 1 for background
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_model_classes)
    return model

# evaluate_model and generate_coco_eval remain largely the same,
# but they will use SCORE_THRESHOLD and IOU_THRESHOLD from current_run_config
# for the custom F1 calculation in evaluate_model.

def evaluate_model(model, data_loader, device, current_run_config):
    # This function calculates custom Precision, Recall, F1
    # It will use current_run_config['SCORE_THRESHOLD'] and current_run_config['IOU_THRESHOLD']
    # for its internal P/R/F1 calculation that drives the training loop.
    
    # Ensure these specific thresholds are used for this evaluation
    score_threshold_eval = current_run_config.get('SCORE_THRESHOLD_EVAL_F1', current_run_config['SCORE_THRESHOLD'])
    iou_threshold_eval = current_run_config.get('IOU_THRESHOLD_EVAL_F1', current_run_config['IOU_THRESHOLD'])
    logging.debug(f"evaluate_model (custom F1) using ScoreThr: {score_threshold_eval}, IoUThr: {iou_threshold_eval}")


    num_actual_classes = len(current_run_config['CLASS_NAMES'])
    # Initialize TP, FP, FN for each class (1-indexed)
    tp = [0] * (num_actual_classes + 1)
    fp = [0] * (num_actual_classes + 1)
    fn = [0] * (num_actual_classes + 1)
    
    model.eval()
    inf_times = []
    per_img_inf_times = []

    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            
            batch_start_time = time.time()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_dt = time.time() - batch_start_time
            inf_times.append(batch_dt)
            per_img_inf_times.extend([batch_dt / len(images)] * len(images))

            for i in range(len(outputs)):
                output = outputs[i]
                target = targets[i] # Assuming target is already on CPU from dataloader or collate_fn

                gt_boxes = target['boxes'].cpu().numpy()
                gt_labels = target['labels'].cpu().numpy() # These are 1-indexed

                pred_scores = output['scores'].cpu().numpy()
                pred_boxes = output['boxes'].cpu().numpy()
                pred_labels = output['labels'].cpu().numpy() # These are 1-indexed

                # Filter predictions by score_threshold_eval
                keep = pred_scores >= score_threshold_eval
                filtered_pred_boxes = pred_boxes[keep]
                filtered_pred_labels = pred_labels[keep]
                # filtered_pred_scores = pred_scores[keep] # If needed

                # Match predictions to ground truths
                matched_gt_indices_for_image = set()

                for p_box, p_label in zip(filtered_pred_boxes, filtered_pred_labels):
                    # Ensure predicted label is valid (1 to num_actual_classes)
                    if not (1 <= p_label <= num_actual_classes):
                        # This case should ideally not happen if model is trained correctly,
                        # but good to guard against. Could be counted as FP for an 'unknown' class if desired.
                        continue 
                    
                    best_iou_for_pred = 0
                    best_gt_idx_for_pred = -1

                    for gt_idx, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if g_label != p_label: # Must be same class
                            continue
                        if gt_idx in matched_gt_indices_for_image: # GT already matched
                            continue
                        
                        iou = compute_iou(p_box, g_box)
                        if iou > best_iou_for_pred:
                            best_iou_for_pred = iou
                            best_gt_idx_for_pred = gt_idx
                    
                    if best_iou_for_pred >= iou_threshold_eval: # Using iou_threshold_eval
                        if best_gt_idx_for_pred != -1: # Should always be true if best_iou > 0
                            tp[p_label] += 1
                            matched_gt_indices_for_image.add(best_gt_idx_for_pred)
                    else: # Prediction did not match any GT with sufficient IoU or was wrong class (already filtered)
                        fp[p_label] += 1 
                
                # Calculate False Negatives for this image
                for gt_idx, g_label in enumerate(gt_labels):
                    if not (1 <= g_label <= num_actual_classes): # Should not happen with clean GT
                        continue
                    if gt_idx not in matched_gt_indices_for_image:
                        fn[g_label] += 1
                        
    class_metrics = {}
    overall_tp = 0
    overall_fp = 0
    overall_fn = 0

    for i, class_name in enumerate(current_run_config['CLASS_NAMES']):
        class_label_idx = i + 1 # CLASS_NAMES[0] is label 1, etc.
        c_tp = tp[class_label_idx]
        c_fp = fp[class_label_idx]
        c_fn = fn[class_label_idx]

        precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        recall = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        class_metrics[class_name] = {'precision': precision, 'recall': recall, 'f1': f1, 
                                     'tp': c_tp, 'fp': c_fp, 'fn': c_fn}
        overall_tp += c_tp
        overall_fp += c_fp
        overall_fn += c_fn

    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0.0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0.0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
    
    inf_total_batch_time = sum(inf_times)
    inf_avg_per_img = np.mean(per_img_inf_times) if per_img_inf_times else 0.0
    # Total images = len(data_loader.dataset)
    fps = len(data_loader.dataset) / inf_total_batch_time if inf_total_batch_time > 0 else 0.0
    
    overall_summary = {
        'precision': overall_precision, 
        'recall': overall_recall, 
        'f1': overall_f1, 
        'tp': overall_tp, 'fp': overall_fp, 'fn': overall_fn,
        'inference': {'total_time': inf_total_batch_time, 
                      'avg_time_per_image': inf_avg_per_img, 
                      'fps': fps}
    }
    return {'overall': overall_summary, 'per_class': class_metrics}


def generate_coco_eval(model, data_loader, device, current_run_config, run_folder_path):
    # This function calculates COCO mAP metrics
    # It uses current_run_config['COCO_EVAL_MIN_SCORE_THRESHOLD'] for filtering detections
    # COCO's internal IoU thresholds are used (0.5 for mAP@.50, 0.5:0.05:0.95 for main mAP)
    logging.debug(f"generate_coco_eval using COCO_EVAL_MIN_SCORE_THRESHOLD: {current_run_config.get('COCO_EVAL_MIN_SCORE_THRESHOLD', 0.01)}")

    model.eval()
    coco_gt_dict = {"images": [], "annotations": [], "categories": []}
    coco_dt_list = [] # List for detections

    # Define categories for COCO format (0-indexed for COCO, but your labels are 1-indexed)
    for i, name in enumerate(current_run_config['CLASS_NAMES']):
        # COCO category IDs should be what the model outputs (1-indexed if that's the case)
        # or mapped if model outputs 0-indexed for classes and pycocotools needs specific IDs.
        # If model outputs 1...N for classes, and your CLASS_NAMES is N long:
        coco_gt_dict["categories"].append({"id": i + 1, "name": name, "supercategory": "object"})


    annotation_id_counter = 1 
    coco_image_id_counter = 0 # This will be 0 to len(dataset)-1

    with torch.no_grad():
        for images, targets in data_loader: # images are lists of tensors, targets are lists of dicts
            images_on_device = [img.to(device) for img in images]
            outputs = model(images_on_device) # List of dicts, one per image

            for i in range(len(images)): # Process each image in the batch
                img_tensor_cpu = images[i] # Original tensor on CPU for shape
                target = targets[i]      # Corresponding target dict
                output = outputs[i]      # Corresponding output dict

                # Use original dataset index if available, otherwise use a running counter
                # The target['image_id'] from AbsoluteDataset is already 0-indexed batch-wise index.
                # For COCO, image_id should be consistent for an image across GT and DT.
                # We'll use a running coco_image_id_counter that matches the image's iteration.
                
                current_coco_image_id = target['image_id'].item() # This is the original dataset index
                                
                # Add image info to COCO GT
                # Only add if not already added (important due to batching)
                # A simple way is to use a set, or build image_infos first, then gt/dt
                # For now, assuming one pass through dataloader for eval
                # The structure of generate_coco_eval from previous version was better here,
                # creating image entries once. Let's adapt.
                # This loop structure assumes batch_size=1 for unique image_id processing per iteration.
                # If batch_size > 1, this needs care.
                # Given data_loader for eval is usually batch_size=1, this might be fine.
                # The target['image_id'] is the index from ValidationDataset

                # We need unique image IDs for COCO format.
                # The image_id from target is its index in the dataset.
                # We will map these to a sequential coco_image_id if needed, or just use them.
                # For simplicity, let's use target['image_id'].item() as the coco image id.
                
                # This check is problematic if batch_size > 1, image info would be added multiple times.
                # This function is called once per validation epoch.
                # It's better to build image_infos separately or ensure unique IDs.
                # Let's assume unique image_ids are handled by `target['image_id']`
                
                # This adds image info for every image in every batch - if an image appears
                # in multiple target dicts (which it won't with shuffle=False and one epoch eval),
                # it would be an issue. Here, it should be fine.
                coco_gt_dict["images"].append({
                    "id": current_coco_image_id, 
                    # "original_dataset_id": current_coco_image_id, # Redundant if using it as main id
                    "width": img_tensor_cpu.shape[2], # W
                    "height": img_tensor_cpu.shape[1] # H
                })

                # Ground Truth Annotations for COCO
                gt_boxes_cpu = target['boxes'].cpu().numpy()
                gt_labels_cpu = target['labels'].cpu().numpy() # Assumed 1-indexed

                for box, model_label_id in zip(gt_boxes_cpu, gt_labels_cpu):
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    # Ensure label_id is int for JSON
                    coco_gt_dict["annotations"].append({
                        "id": annotation_id_counter,
                        "image_id": current_coco_image_id,
                        "category_id": int(model_label_id), # Use 1-indexed model label directly
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "area": float(width * height),
                        "iscrowd": 0
                    })
                    annotation_id_counter += 1

                # Detections for COCO
                pred_boxes_cpu = output["boxes"].cpu().numpy()
                pred_scores_cpu = output["scores"].cpu().numpy()
                pred_labels_cpu = output["labels"].cpu().numpy() # Assumed 1-indexed

                min_score_thresh = current_run_config.get('COCO_EVAL_MIN_SCORE_THRESHOLD', 0.01)
                
                for box, score, model_label_id in zip(pred_boxes_cpu, pred_scores_cpu, pred_labels_cpu):
                    if score < min_score_thresh:
                        continue
                    x1, y1, x2, y2 = box
                    width = x2 - x1
                    height = y2 - y1
                    # Ensure model_label_id is int for JSON
                    coco_dt_list.append({
                        "image_id": current_coco_image_id,
                        "category_id": int(model_label_id), # Use 1-indexed model label directly
                        "bbox": [float(x1), float(y1), float(width), float(height)],
                        "score": float(score)
                    })
                
                # coco_image_id_counter +=1 # This was from a previous version and is not needed if using target['image_id']

    if not coco_gt_dict["annotations"]:
        logging.warning("No ground truth annotations found for COCO evaluation. Returning zero metrics.")
        return {"mAP@.50": 0.0, "mAP@.75": 0.0, "mAP@.50:95": 0.0, "info": "No GT for COCO eval"}
    if not coco_dt_list:
        logging.warning(f"No detections found after score thresholding ({min_score_thresh}) for COCO evaluation. Returning zero metrics.")
        return {"mAP@.50": 0.0, "mAP@.75": 0.0, "mAP@.50:95": 0.0, "info": f"No Detections > {min_score_thresh}"}

    # Create COCO ground truth object
    # Need to ensure image IDs are unique in coco_gt_dict["images"]
    # A robust way:
    unique_images_dict = {img['id']: img for img in coco_gt_dict["images"]}
    coco_gt_dict["images"] = list(unique_images_dict.values())
    
    # Temporary save and load for COCO API (can be slow for very large datasets)
    # Consider in-memory COCO object creation if performance is an issue
    gt_temp_path = os.path.join(run_folder_path, "_temp_coco_gt.json")
    with open(gt_temp_path, 'w') as f:
        json.dump(coco_gt_dict, f)
    
    coco_gt_obj = COCO(gt_temp_path)
    coco_dt_obj = coco_gt_obj.loadRes(coco_dt_list)
    
    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, iouType='bbox')
    # coco_eval.params.iouThrs = np.array([0.5]) # For mAP@.50 only if needed, but default is 0.5:0.05:0.95
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    summary_file_path = os.path.join(run_folder_path, 'coco_evaluation_summary.txt')
    original_stdout = sys.stdout
    with open(summary_file_path, 'w') as f_summary:
        sys.stdout = f_summary
        coco_eval.summarize()
    sys.stdout = original_stdout
    logging.info(f"COCO evaluation summary saved to: {summary_file_path}")
    # with open(summary_file_path, 'r') as f_summary_read: # Log summary to console
    #     logging.info("COCO Summary:\n" + f_summary_read.read())

    stats = coco_eval.stats # mAP@.50:95, mAP@.50, mAP@.75, ARs etc.
    metrics = {
        "mAP@.50": float(stats[1]) if len(stats) > 1 else 0.0,
        "mAP@.75": float(stats[2]) if len(stats) > 2 else 0.0,
        "mAP@.50:95": float(stats[0]) if len(stats) > 0 else 0.0,
        "AR@1": float(stats[6]) if len(stats) > 6 else 0.0,
        "AR@10": float(stats[7]) if len(stats) > 7 else 0.0,
        "AR@100": float(stats[8]) if len(stats) > 8 else 0.0,
    }

    # Per-class AP@0.50 (using the first IoU threshold which is 0.5)
    # This requires re-running parts of eval or careful indexing if params change.
    # A simpler way is to iterate through catIds AFTER the main accumulate
    per_class_ap_at_50 = {}
    for cat_idx, cat_info in enumerate(coco_gt_dict["categories"]):
        cat_id = cat_info['id'] # This is the 1-indexed class ID
        cat_name = cat_info['name']
        
        # Create a new COCOeval object for per-class to avoid state issues
        coco_eval_per_class = COCOeval(coco_gt_obj, coco_dt_obj, iouType='bbox')
        coco_eval_per_class.params.catIds = [cat_id]
        # coco_eval_per_class.params.iouThrs = np.array([0.5]) # For AP@.50 specifically
        coco_eval_per_class.evaluate()
        coco_eval_per_class.accumulate()
        # coco_eval_per_class.summarize() # Optionally print per-class summary
        
        # stats_per_class[0] would be mAP@.50:.05:.95 for this class
        # stats_per_class[1] would be mAP@.50 for this class
        # If iouThrs was set to just [0.5], then stats_per_class[0] is AP@.50
        # To get AP@.50 robustly from default iouThrs (0.5:0.05:0.95):
        # Precision array: [T, R, K, A, M] T=iouThrs, R=recallThrs, K=catIds, A=areaRng, M=maxDets
        # We need K for current cat_id, T for iouThr=0.5
        
        # This part from the original script for per-class AP was more direct using `coco_eval.eval['precision']`
        # Let's use that approach with the main coco_eval object
        precisions_data = coco_eval.eval["precision"] 
        # Find index for IoU=0.5 (usually the first one if default iouThrs are used)
        iou_idx_for_0_50 = np.where(np.isclose(coco_eval.params.iouThrs, 0.5))[0]
        if len(iou_idx_for_0_50) == 0: # Fallback if 0.5 not exactly present
            iou_idx_for_0_50 = 0 
        else:
            iou_idx_for_0_50 = iou_idx_for_0_50[0]

        # Find index for this category_id in the evaluation object's category list
        # coco_eval.params.catIds contains all category IDs used in the current eval round.
        # If we ran summarize() on all classes, then K is the index in this list.
        # Need to map our cat_id (e.g. 1, 2, ...) to the K index (0, 1, ...)
        # This is tricky if not all classes from 1 to N are present.
        # A safer way: use the K_idx from enumerate(current_run_config['CLASS_NAMES'])
        # where cat_id = k_idx + 1
        
        k_idx_for_eval = -1
        for eval_k_idx, eval_cat_id in enumerate(coco_eval.params.catIds):
            if eval_cat_id == cat_id:
                k_idx_for_eval = eval_k_idx
                break
        
        if k_idx_for_eval != -1:
            # precision_values_for_class_ap50: [Recall thresholds]
            precision_values_for_class_ap50 = precisions_data[iou_idx_for_0_50, :, k_idx_for_eval, 0, 2] # Area=all, MaxDets=100
            ap_val = np.mean(precision_values_for_class_ap50[precision_values_for_class_ap50 > -1]) 
            if np.isnan(ap_val): ap_val = 0.0
            per_class_ap_at_50[cat_name] = float(ap_val)
        else:
            per_class_ap_at_50[cat_name] = 0.0 # Class not found in eval params somehow

    metrics["per_class_ap_at_0.50"] = per_class_ap_at_50
    with open(os.path.join(run_folder_path, "per_class_ap_at_0.50_coco.json"), 'w') as f:
        json.dump(per_class_ap_at_50, f, indent=2)

    # Clean up temporary JSON file for GT
    if os.path.exists(gt_temp_path):
        os.remove(gt_temp_path)
        
    return metrics


def train_model_with_variation(variation_idx, variation_params): # REMOVED output_base_dir_override
    current_run_config = {**BASE_CONFIG, **variation_params} # Merge base with variation specifics

    # --- Parameter Validation and Setup ---
    # ... (rest of your parameter validation and NUM_CLASSES_MODEL setup remains the same) ...
    if 'CLASS_NAMES' not in current_run_config or not current_run_config['CLASS_NAMES']:
        temp_logger = logging.getLogger(); 
        if not temp_logger.hasHandlers(): logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
        logging.error(f"Variation {variation_idx+1} ('{current_run_config.get('VARIATION_NAME', 'Unnamed')}'): CLASS_NAMES missing or empty. Skipping."); return
    
    current_run_config['NUM_CLASSES_MODEL'] = len(current_run_config['CLASS_NAMES']) + 1 
    seed_everything(current_run_config['SEED'])
    
    # --- Run Naming and Folder Setup ---
    variation_name_str = current_run_config.get("VARIATION_NAME", f"Run_{variation_idx+1}_Generic")
    opt_type_for_name = current_run_config.get("OPTIMIZER_TYPE", "AdamW")
    if opt_type_for_name.lower() not in variation_name_str.lower(): # Add optimizer if not in name
        variation_name_str = f"{variation_name_str}_{opt_type_for_name}"
    
    run_name = f"{variation_name_str}_LR{current_run_config['LEARNING_RATE']}"
    if current_run_config.get("USE_GRADIENT_ACCUMULATION", False):
        run_name += f"_EffBS{current_run_config.get('TARGET_EFFECTIVE_BATCH_SIZE', 'NA')}"
    else:
        run_name += f"_BS{current_run_config['BATCH_SIZE']}"
    run_name += f"_ImgSz{current_run_config['TARGET_SIZE'][0]}"
    
    # Use OUTPUT_BASE_DIR from current_run_config
    output_base_dir = current_run_config['OUTPUT_BASE_DIR'] 
    run_folder_path = os.path.join(output_base_dir, run_name)
    os.makedirs(run_folder_path, exist_ok=True) # This will create the base dir too if it doesn't exist


    # --- Logging Setup ---
    root_logger = logging.getLogger(); 
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    log_file_path = os.path.join(run_folder_path, 'training_run.log')
    logging.basicConfig(level=logging.DEBUG if current_run_config.get("DEBUG_LOGGING", False) else logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
    
    logging.info(f"===== Starting Training Run: {run_name} =====")
    logging.info(f"Outputting to folder: {run_folder_path}")
    logging.info(f"Full Run Configuration: {json.dumps(current_run_config, indent=2, sort_keys=True)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); logging.info(f"Using device: {device}")

    # --- Dataset and DataLoader Setup ---
    try:
        # Using AbsoluteDataset, which handles resizing internally
        train_ds = AbsoluteDataset(images_dir=current_run_config['TRAIN_IMAGE_DIR'], 
                                   labels_dir=current_run_config['TRAIN_LABEL_DIR'],
                                   target_image_size=current_run_config['TARGET_SIZE'], 
                                   class_names_for_this_dataset=current_run_config['CLASS_NAMES'])
        val_ds = AbsoluteDataset(images_dir=current_run_config['VAL_IMAGE_DIR'], 
                                 labels_dir=current_run_config['VAL_LABEL_DIR'],
                                 target_image_size=current_run_config['TARGET_SIZE'], 
                                 class_names_for_this_dataset=current_run_config['CLASS_NAMES'])
    except FileNotFoundError as e: logging.error(f"Dataset directory not found: {e}. Skipping run."); return
    except Exception as e: logging.error(f"Error initializing dataset: {e}. Skipping run."); return

    if not len(train_ds): logging.error(f"Training dataset is empty. Path: {current_run_config['TRAIN_IMAGE_DIR']}. Skipping run."); return
    if not len(val_ds): logging.warning(f"Validation dataset is empty. Path: {current_run_config['VAL_IMAGE_DIR']}. Validation metrics will be zero.")

    train_loader = DataLoader(train_ds, batch_size=current_run_config['BATCH_SIZE'], shuffle=True, 
                              collate_fn=collate_fn, num_workers=current_run_config.get('NUM_WORKERS', 4), pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=current_run_config['BATCH_SIZE'], shuffle=False, # Eval usually uses BS=1 or same as train
                            collate_fn=collate_fn, num_workers=current_run_config.get('NUM_WORKERS', 4), pin_memory=True)
    logging.info(f"Training dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}")
    logging.info(f"Number of object classes: {len(current_run_config['CLASS_NAMES'])} -> Model configured for {current_run_config['NUM_CLASSES_MODEL']} outputs (incl. background).")

    # --- Model Setup ---
    model = get_model(num_model_classes=current_run_config['NUM_CLASSES_MODEL']).to(device)

    # --- Optimizer Setup ---
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer_type = current_run_config.get("OPTIMIZER_TYPE", "AdamW").upper()
    lr = current_run_config['LEARNING_RATE']
    wd = current_run_config.get('WEIGHT_DECAY', 0.0001) # Default WD

    if optimizer_type == "SGD":
        momentum = current_run_config.get('SGD_MOMENTUM', 0.9)
        optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=wd)
        logging.info(f"Using SGD optimizer with LR={lr}, Momentum={momentum}, WeightDecay={wd}")
    else: # Default to AdamW
        betas = current_run_config.get('ADAMW_BETAS', (0.9, 0.999))
        optimizer = optim.AdamW(params, lr=lr, betas=betas, weight_decay=wd)
        logging.info(f"Using AdamW optimizer with LR={lr}, Betas={betas}, WeightDecay={wd}")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, 
                                                         patience=current_run_config.get('SCHEDULER_PATIENCE', 10), 
                                                         verbose=True)
    
    # --- Gradient Accumulation Setup ---
    physical_batch_size = current_run_config['BATCH_SIZE']
    if current_run_config.get("USE_GRADIENT_ACCUMULATION", False):
        target_effective_batch_size = current_run_config.get("TARGET_EFFECTIVE_BATCH_SIZE", physical_batch_size)
        accum_steps = max(1, target_effective_batch_size // physical_batch_size)
    else:
        accum_steps = 1
    effective_batch_size_calculated = physical_batch_size * accum_steps
    logging.info(f"Physical Batch Size: {physical_batch_size}, Gradient Accumulation Steps: {accum_steps}, Effective Batch Size: {effective_batch_size_calculated}")


    # --- Training Loop Initialization ---
    best_metric_val = -1.0  
    best_epoch = 0
    epochs_no_improve = 0
    metrics_log_list = []
    best_metric_name_overall = "N/A"
    warmup_epochs = current_run_config.get("WARMUP_EPOCHS", 0) # Default to 0 if not specified
    initial_lr_target = current_run_config['LEARNING_RATE'] # Target LR after warmup

    for epoch in range(current_run_config['NUM_EPOCHS']):
        # --- LR Warmup ---
        if epoch < warmup_epochs:
            # Linear warmup from a very small value to the initial_lr_target
            # Could also warm up from initial_lr_target / warmup_epochs to initial_lr_target
            # Let's do: start near 0, ramp up to initial_lr_target
            warmup_lr_val = initial_lr_target * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr_val
            logging.info(f"Epoch {epoch+1}/{current_run_config['NUM_EPOCHS']} (Warmup): LR set to {warmup_lr_val:.2e}")
        elif epoch == warmup_epochs and warmup_epochs > 0: # Set to target LR just after warmup finishes
            for param_group in optimizer.param_groups:
                param_group['lr'] = initial_lr_target
            logging.info(f"Epoch {epoch+1}/{current_run_config['NUM_EPOCHS']}: Warmup complete. LR set to {initial_lr_target:.2e}")
        
        # --- Training Phase ---
        model.train()
        epoch_total_loss = 0.0
        # optimizer.zero_grad() # Moved inside accumulation loop logic

        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx % accum_steps == 0: # Zero grad at the start of an accumulation cycle
                optimizer.zero_grad()

            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            try:
                loss_dict = model(images, targets)
                if not loss_dict or not any(torch.is_tensor(l) for l in loss_dict.values()):
                    logging.warning(f"E{epoch+1} B{batch_idx+1}: Invalid/empty loss_dict. Skipping batch."); continue
                loss = sum(l for l in loss_dict.values() if torch.is_tensor(l))
            except Exception as e:
                logging.error(f"E{epoch+1} B{batch_idx+1} Fwd/Loss Error: {e}", exc_info=True); 
                logging.debug(f"Problematic Targets: {targets}"); continue
            
            if torch.isnan(loss) or torch.isinf(loss):
                logging.warning(f"E{epoch+1} B{batch_idx+1}: NaN or Inf loss detected ({loss.item()}). Skipping batch update."); continue

            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            epoch_total_loss += loss.item() # Accumulate non-scaled loss for logging

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Potentially clip gradients here if needed: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                # optimizer.zero_grad() # Zeroed at the start of the next accumulation cycle or after epoch if not accumulating
        
        if accum_steps == 1 : # If not accumulating, ensure grads are zeroed after epoch if not fully divisible
             optimizer.zero_grad()


        avg_epoch_loss = epoch_total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # --- Validation Phase ---
        current_f1, current_map50, current_map50_95 = 0.0, 0.0, 0.0
        val_p, val_r = 0.0, 0.0
        custom_eval_metrics = {'overall': {'f1':0, 'precision':0, 'recall':0}}
        coco_eval_metrics = {} # Will store mAP values

        if val_loader and len(val_loader) > 0 and len(val_ds) > 0:
            # Pass current_run_config to evaluate_model so it can use SCORE_THRESHOLD and IOU_THRESHOLD
            custom_eval_metrics = evaluate_model(model, val_loader, device, current_run_config)
            coco_eval_metrics = generate_coco_eval(model, val_loader, device, current_run_config, run_folder_path)
            
            current_f1 = custom_eval_metrics['overall']['f1']
            val_p = custom_eval_metrics['overall']['precision']
            val_r = custom_eval_metrics['overall']['recall']
            current_map50 = coco_eval_metrics.get("mAP@.50", 0.0)
            current_map50_95 = coco_eval_metrics.get('mAP@.50:95', 0.0)
        else:
            logging.warning(f"E{epoch+1}: Skipping validation (empty val loader/ds). Metrics will be zero.")

        # --- Metric for Scheduler & Best Model (F1 primary, mAP@.50 fallback) ---
        # The F1 here is calculated using current_run_config['SCORE_THRESHOLD'] and current_run_config['IOU_THRESHOLD']
        metric_for_scheduler = current_f1 if current_f1 > 0.001 else (current_map50 if current_map50 > 0.001 else 0.0)
        # Scheduler step should only happen after warmup if warmup is active
        if epoch >= warmup_epochs:
            scheduler.step(metric_for_scheduler)
            logging.debug(f"E{epoch+1}: Scheduler stepped with {'F1' if current_f1 > 0.001 else 'mAP@.50'}: {metric_for_scheduler:.4f}")
        else:
            logging.debug(f"E{epoch+1}: In warmup, scheduler.step not called.")


        logging.info(f"[E{epoch+1}/{current_run_config['NUM_EPOCHS']}] TrainLoss: {avg_epoch_loss:.4f} | Val F1(S:{current_run_config['SCORE_THRESHOLD']}/I:{current_run_config['IOU_THRESHOLD']}): {current_f1:.4f} (P:{val_p:.3f} R:{val_r:.3f}) | mAP@.50:{current_map50:.4f} | mAP@.50-.95:{current_map50_95:.4f} | LR:{optimizer.param_groups[0]['lr']:.2e}")
        
        epoch_log_entry = {
            'epoch': epoch + 1, 'avg_train_loss': avg_epoch_loss,
            'val_custom_eval': custom_eval_metrics, 
            'val_coco_eval': coco_eval_metrics,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        metrics_log_list.append(epoch_log_entry)
        with open(os.path.join(run_folder_path, 'metrics_log_every_epoch.json'), 'w') as f:
            json.dump(metrics_log_list, f, indent=2)

        current_best_candidate_metric = current_f1 if current_f1 > 0.0 else (current_map50 if current_map50 > 0.0 else -1.0)
        chosen_metric_name_for_epoch = "F1" if current_f1 > 0.0 else ("mAP@.50" if current_map50 > 0.0 else "N/A")
        
        if current_best_candidate_metric > best_metric_val:
            best_metric_val = current_best_candidate_metric
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_metric_name_overall = chosen_metric_name_for_epoch
            
            torch.save(model.state_dict(), os.path.join(run_folder_path, 'best_model.pth'))
            logging.info(f"<<< Best Model Saved! Epoch: {best_epoch}, Metric ({best_metric_name_overall}): {best_metric_val:.4f}")
            
            summary_data = {
                'best_epoch': best_epoch,
                'best_metric_name': best_metric_name_overall,
                'best_metric_value': best_metric_val,
                'final_custom_eval_at_best': custom_eval_metrics,
                'final_coco_eval_at_best': coco_eval_metrics,
                'config_summary': {k: v for k, v in current_run_config.items() 
                                   if not (isinstance(v, list) and len(v) > 10)} # Avoid overly long lists in summary
            }
            with open(os.path.join(run_folder_path, 'best_metrics_summary.json'), 'w') as f:
                json.dump(summary_data, f, indent=2)
        else:
            epochs_no_improve += 1
            logging.info(f"E{epoch+1}: Metric ({chosen_metric_name_for_epoch}): {current_best_candidate_metric:.4f} did not improve from best ({best_metric_name_overall}): {best_metric_val:.4f}. No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= current_run_config.get('EARLY_STOPPING_PATIENCE', 30): # Use get with default
            logging.info(f"===== Early stopping triggered at epoch {epoch+1}. {best_metric_name_overall} did not improve from {best_metric_val:.4f} for {epochs_no_improve} consecutive epochs. =====")
            break
            
    logging.info(f"===== Finished Training Run: {run_name}. Best model from Epoch {best_epoch} with {best_metric_name_overall}: {best_metric_val:.4f}. Log files saved in: {run_folder_path} =====")


if __name__ == '__main__':
    # Get the base output directory from BASE_CONFIG to ensure it's created once
    # If a variation overrides OUTPUT_BASE_DIR, its specific base will be created by os.makedirs inside the loop
    top_level_runs_dir = BASE_CONFIG.get("OUTPUT_BASE_DIR", "Controlled_FasterRCNN_Runs_F1Opt_Default") # Fallback if not in BASE_CONFIG
    os.makedirs(top_level_runs_dir, exist_ok=True) 
    
    # Default SCORE_THRESHOLD and IOU_THRESHOLD for F1 optimization are now in BASE_CONFIG
    # TRAINING_VARIATIONS can still override them if needed for a specific run.

    for i, variation_params in enumerate(TRAINING_VARIATIONS):
        if "VARIATION_NAME" not in variation_params:
            # ... (your default VARIATION_NAME generation logic) ...
            lr_str = variation_params.get('LEARNING_RATE', 'NA')
            opt_str= variation_params.get('OPTIMIZER_TYPE', 'OptNA')
            bs_str = variation_params.get('BATCH_SIZE', 'NA')
            sz_tuple = variation_params.get('TARGET_SIZE', ('NA','NA'))
            sz_str = sz_tuple[0] if isinstance(sz_tuple, tuple) and len(sz_tuple) > 0 else 'NA'
            variation_params["VARIATION_NAME"] = f"Run_{i+1}_{opt_str}_LR{lr_str}_BS{bs_str}_SZ{sz_str}"
            
        print(f"\n{'='*40}\nProcessing Variation {i+1}/{len(TRAINING_VARIATIONS)}: {variation_params['VARIATION_NAME']}\n{'='*40}")
        train_model_with_variation(i, variation_params) # REMOVED output_base_dir_override
        print(f"\n{'='*40}\nFinished Variation {i+1}/{len(TRAINING_VARIATIONS)}: {variation_params['VARIATION_NAME']}\n{'='*40}\n")
        
    print("All training variations processed successfully.")