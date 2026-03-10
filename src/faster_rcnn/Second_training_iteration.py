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

# === BASE CONFIGURATION ===
BASE_CONFIG = {
    "EFFECTIVE_BATCH": 64,
    "NUM_EPOCHS": 400,
    "SCORE_THRESHOLD": 0.4, # Used for custom F1 calculation in evaluate_model
    "COCO_EVAL_MIN_SCORE_THRESHOLD": 0.01, # Used for filtering detections before COCO eval
    "IOU_THRESHOLD": 0.6, # Used for custom F1 calculation in evaluate_model
    "EARLY_STOPPING_PATIENCE": 30,
    "NUM_WORKERS": 4,
    "SEED": 42,
    "SCHEDULER_PATIENCE": 10,
    "DEBUG_LOGGING": False,

    "TRAIN_IMAGE_DIR": "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/images",
    "TRAIN_LABEL_DIR": "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/labels_minmax",
    "VAL_IMAGE_DIR":    "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/val/images",
    "VAL_LABEL_DIR":    "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/val/labels_minmax",
    "CLASS_NAMES": ["Tripos longipes", "Tripos fusus", "Tripos lineatum/furca", "Chaetoceros", "Coscinodiscus"], # 5 classes
}

# === TRAINING VARIATIONS ===
SINGLE_CLASS_BASE_PATH = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318"
SINGLE_CLASS_NAME_LIST = ["Plankton"]

TRAINING_VARIATIONS = [
    {
        "VARIATION_NAME": "MultiClass_640_LR0.0002_DefaultPaths_mAP50Optimized", # Name updated for clarity
        "TARGET_SIZE": (640, 640), "BATCH_SIZE": 4, "LEARNING_RATE": 0.0002,
    },
    # { # SINGLE CLASS RUN COMMENTED OUT
    #     "VARIATION_NAME": "SingleClass_640_LR0.0002",
    #     "TARGET_SIZE": (640, 640), "BATCH_SIZE": 4, "LEARNING_RATE": 0.0002,
    #     "CLASS_NAMES": SINGLE_CLASS_NAME_LIST,
    #     "TRAIN_IMAGE_DIR": os.path.join(SINGLE_CLASS_BASE_PATH, "train/images"),
    #     "TRAIN_LABEL_DIR": os.path.join(SINGLE_CLASS_BASE_PATH, "train/labels_minmax"),
    #     "VAL_IMAGE_DIR":   os.path.join(SINGLE_CLASS_BASE_PATH, "val/images"),
    #     "VAL_LABEL_DIR":   os.path.join(SINGLE_CLASS_BASE_PATH, "val/labels_minmax"),
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

class AbsoluteDataset(Dataset):
    def __init__(self, images_dir, labels_dir, target_image_size, class_names_for_this_dataset):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.target_width, self.target_height = target_image_size
        self.class_names = class_names_for_this_dataset
        self.images = []
        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not os.path.isdir(self.labels_dir):
            raise FileNotFoundError(f"Labels directory not found: {self.labels_dir}")

        for f_name in sorted(os.listdir(images_dir)):
            if f_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                self.images.append(f_name)
        if not self.images:
            logging.warning(f"No images found in {images_dir}. Please check path and extensions.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            logging.error(f"Failed to load image: {img_path}. Using black placeholder.")
            img = np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_height, original_width = img.shape[:2]
        img_resized = cv2.resize(img, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

        label_path = os.path.join(self.labels_dir, os.path.splitext(img_file)[0] + '.txt')
        boxes, labels = [], []
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
                    if len(float_parts) != 5:
                        logging.warning(f"L{line_num+1} in {label_path}: Expected 5 parts, got {len(float_parts)} ('{line.strip()}'). Skipping.")
                        continue

                    file_cls_id_from_file = int(float_parts[0])
                    if not (1 <= file_cls_id_from_file <= len(self.class_names)):
                        logging.warning(f"Class ID {file_cls_id_from_file} from file {label_path} (line {line_num+1}) "
                                        f"is out of expected range [1, {len(self.class_names)}] for "
                                        f"defined classes: {self.class_names}. Skipping annotation.")
                        continue
                    model_label = file_cls_id_from_file

                    center_x_orig, center_y_orig, w_orig, h_orig = float_parts[1], float_parts[2], float_parts[3], float_parts[4]
                    if original_width == 0 or original_height == 0:
                        logging.error(f"Img {img_file}: zero original dim. Skipping box scaling.")
                        continue

                    x_scale = self.target_width / original_width
                    y_scale = self.target_height / original_height
                    x1_orig = center_x_orig - w_orig / 2; y1_orig = center_y_orig - h_orig / 2
                    x2_orig = center_x_orig + w_orig / 2; y2_orig = center_y_orig + h_orig / 2
                    x1_s = x1_orig * x_scale; y1_s = y1_orig * y_scale
                    x2_s = x2_orig * x_scale; y2_s = y2_orig * y_scale
                    x1_s = max(0.0, x1_s); y1_s = max(0.0, y1_s)
                    x2_s = min(float(self.target_width - 1e-3), x2_s)
                    y2_s = min(float(self.target_height - 1e-3), y2_s)

                    if x2_s > x1_s and y2_s > y1_s:
                        boxes.append([x1_s, y1_s, x2_s, y2_s])
                        labels.append(model_label)

        img_tensor = TF.to_tensor(img_resized.copy())
        target = {"image_id": torch.tensor([idx], dtype=torch.int64)}
        if boxes:
            target_boxes = torch.tensor(boxes, dtype=torch.float32)
            target["boxes"] = target_boxes
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
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

def get_model(num_model_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_model_classes)
    return model

def evaluate_model(model, data_loader, device, current_run_config): # Custom F1/P/R
    num_actual_classes = len(current_run_config['CLASS_NAMES'])
    tp = [0] * (num_actual_classes + 1); fp = [0] * (num_actual_classes + 1); fn = [0] * (num_actual_classes + 1)
    model.eval(); inf_times = []; per_img_inf_times = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            start_time = time.time(); outputs = model(images)
            if device.type == 'cuda': torch.cuda.synchronize()
            dt = time.time() - start_time; inf_times.append(dt); per_img_inf_times.extend([dt / len(images)] * len(images))
            for output, target in zip(outputs, targets):
                gt_boxes = target['boxes'].cpu().numpy(); gt_labels = target['labels'].cpu().numpy()
                pred_scores = output['scores'].cpu().numpy(); pred_boxes = output['boxes'].cpu().numpy(); pred_labels = output['labels'].cpu().numpy()
                keep = pred_scores >= current_run_config['SCORE_THRESHOLD']
                pred_boxes, pred_labels = pred_boxes[keep], pred_labels[keep]
                matched_gt_indices = set()
                for p_box, p_label in zip(pred_boxes, pred_labels):
                    if not (1 <= p_label <= num_actual_classes): continue
                    best_iou, best_gt_idx = 0, -1
                    for gt_idx, (g_box, g_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if g_label != p_label or gt_idx in matched_gt_indices: continue
                        iou = compute_iou(p_box, g_box)
                        if iou > best_iou: best_iou, best_gt_idx = iou, gt_idx
                    if best_iou >= current_run_config['IOU_THRESHOLD']:
                        if best_gt_idx != -1: tp[p_label] += 1; matched_gt_indices.add(best_gt_idx)
                    else: fp[p_label] += 1
                for gt_idx, g_label in enumerate(gt_labels):
                    if not (1 <= g_label <= num_actual_classes): continue
                    if gt_idx not in matched_gt_indices: fn[g_label] += 1
    class_metrics = {}; overall_tp, overall_fp, overall_fn = 0, 0, 0
    for i, class_name in enumerate(current_run_config['CLASS_NAMES']):
        class_label_idx = i + 1
        c_tp, c_fp, c_fn = tp[class_label_idx], fp[class_label_idx], fn[class_label_idx]
        precision = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0
        recall = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        class_metrics[class_name] = {'precision': precision, 'recall': recall, 'f1': f1, 'tp': c_tp, 'fp': c_fp, 'fn': c_fn}
        overall_tp += c_tp; overall_fp += c_fp; overall_fn += c_fn
    overall_precision = overall_tp / (overall_tp + overall_fp) if (overall_tp + overall_fp) > 0 else 0
    overall_recall = overall_tp / (overall_tp + overall_fn) if (overall_tp + overall_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    inf_total = sum(inf_times); inf_avg_per_img = np.mean(per_img_inf_times) if per_img_inf_times else 0
    fps = len(per_img_inf_times) / inf_total if inf_total > 0 else 0
    overall_summary = {'precision': overall_precision, 'recall': overall_recall, 'f1': overall_f1, 'tp': overall_tp, 'fp': overall_fp, 'fn': overall_fn,
                       'inference': {'total_time': inf_total, 'avg_time_per_image': inf_avg_per_img, 'fps': fps}}
    return {'overall': overall_summary, 'per_class': class_metrics}

def generate_coco_eval(model, data_loader, device, current_run_config, run_folder_path): # COCO mAP
    model.eval(); coco_gt_dict = {"images": [], "annotations": [], "categories": []}
    coco_dt_list = []
    for i, name in enumerate(current_run_config['CLASS_NAMES']):
        coco_gt_dict["categories"].append({"id": i, "name": name, "supercategory": "object"})
    annotation_id_counter = 1; coco_image_id_counter = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for i in range(len(images)):
                img_tensor = images[i]; original_dataset_idx = targets[i]['image_id'].item()
                coco_gt_dict["images"].append({"id": coco_image_id_counter, "original_dataset_id": original_dataset_idx,
                                               "width": img_tensor.shape[2], "height": img_tensor.shape[1] })
                gt_boxes = targets[i]['boxes'].cpu().numpy()
                gt_model_labels = targets[i]['labels'].cpu().numpy()
                for box, model_label in zip(gt_boxes, gt_model_labels):
                    x1, y1, x2, y2 = box; width = x2 - x1; height = y2 - y1
                    coco_category_id = int(model_label - 1)
                    coco_gt_dict["annotations"].append({"id": annotation_id_counter, "image_id": coco_image_id_counter,
                                                        "category_id": coco_category_id, "bbox": [float(x1), float(y1), float(width), float(height)],
                                                        "area": float(width * height), "iscrowd": 0 }); annotation_id_counter += 1
                pred_boxes = outputs[i]["boxes"].cpu().numpy(); pred_scores = outputs[i]["scores"].cpu().numpy()
                pred_model_labels = outputs[i]["labels"].cpu().numpy()
                coco_min_score_thresh = current_run_config.get('COCO_EVAL_MIN_SCORE_THRESHOLD', 0.01)
                for box, score, model_label in zip(pred_boxes, pred_scores, pred_model_labels):
                    if score < coco_min_score_thresh: continue
                    x1, y1, x2, y2 = box; width = x2 - x1; height = y2 - y1
                    coco_category_id = int(model_label - 1)
                    if not (0 <= coco_category_id < len(current_run_config['CLASS_NAMES'])): continue
                    coco_dt_list.append({"image_id": coco_image_id_counter, "category_id": coco_category_id,
                                         "bbox": [float(x1), float(y1), float(width), float(height)], "score": float(score) })
                coco_image_id_counter += 1
    if not coco_gt_dict["annotations"]:
        logging.warning("No GT for COCO eval. Zero metrics."); return {"mAP@.50": 0, "mAP@.50:95": 0, "info": "No GT"}
    if not coco_dt_list:
        logging.warning(f"No Dets for COCO eval (score th: {coco_min_score_thresh}). Zero metrics."); return {"mAP@.50": 0, "mAP@.50:95": 0, "info": f"No Dets >{coco_min_score_thresh}"}
    
    coco_gt_obj = COCO(); coco_gt_obj.dataset = coco_gt_dict; coco_gt_obj.createIndex()
    coco_dt_obj = coco_gt_obj.loadRes(coco_dt_list); coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, iouType='bbox')
    coco_eval.evaluate(); coco_eval.accumulate()
    
    summary_path = os.path.join(run_folder_path, 'coco_evaluation_summary.txt') # Changed filename
    original_stdout = sys.stdout
    with open(summary_path, 'w') as f_summary:
        sys.stdout = f_summary
        coco_eval.summarize()
    sys.stdout = original_stdout
    logging.info(f"COCO evaluation summary saved to: {summary_path}")
    # with open(summary_path, 'r') as f_summary_read: logging.info("COCO Summary:\n" + f_summary_read.read()) # Optional: print to main log

    stats = coco_eval.stats
    # stats[0] is mAP@.50:.05:.95, stats[1] is mAP@.50
    metrics = {"mAP@.50": stats[1], "mAP@.75": stats[2], "mAP@.50:95": stats[0], 
               "AR@1": stats[6], "AR@10": stats[7], "AR@100": stats[8]}
    
    per_class_ap_at_50 = {} # Specifically for AP@0.50
    precisions_data = coco_eval.eval["precision"]
    # For AP@.50, we use the first IoU threshold index (0.50)
    # Dimensions of precision: [IoU_thresh, recall_thresh, class, area_range, max_detections]
    iou_idx_for_0_50 = 0 # coco_eval.params.iouThrs[0] is 0.5
    area_rng_idx_all = 0 # coco_eval.params.areaRngLbl[0] is 'all'
    max_dets_idx_100 = 2 # coco_eval.params.maxDets[2] is 100
    
    for class_idx, class_name in enumerate(current_run_config['CLASS_NAMES']): # class_idx is 0-indexed COCO category_id
        # Get precision values for specific IoU (0.50), class, area ('all'), maxDets (100)
        precision_values_for_class_ap50 = precisions_data[iou_idx_for_0_50, :, class_idx, area_rng_idx_all, max_dets_idx_100]
        ap_val = np.mean(precision_values_for_class_ap50[precision_values_for_class_ap50 > -1]) # Filter out -1s (no detections for that recall)
        if np.isnan(ap_val): ap_val = 0.0
        per_class_ap_at_50[class_name] = float(ap_val)
        
    metrics["per_class_ap_at_0.50"] = per_class_ap_at_50
    with open(os.path.join(run_folder_path, "per_class_ap_at_0.50_coco.json"), 'w') as f: json.dump(per_class_ap_at_50, f, indent=2) # Changed filename
    return metrics

def train_model_with_variation(variation_idx, variation_params):
    current_run_config = {**BASE_CONFIG, **variation_params}
    if 'CLASS_NAMES' not in current_run_config or not current_run_config['CLASS_NAMES']:
        # Basic logging if main logger not yet set up for this variation
        temp_logger = logging.getLogger()
        if not temp_logger.hasHandlers(): logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
        logging.error(f"Variation {variation_idx+1}: CLASS_NAMES missing or empty. Skipping this variation."); return

    current_run_config['NUM_CLASSES_MODEL'] = len(current_run_config['CLASS_NAMES']) + 1
    seed_everything(current_run_config['SEED'])
    
    variation_name_str = current_run_config.get("VARIATION_NAME", f"Run{variation_idx+1}")
    # Ensure run name reflects mAP@.50 optimization if not already specified
    if "mAP50Optimized" not in variation_name_str and "mAPOptimized" not in variation_name_str: # Check general mAP too
        variation_name_str += "_mAP50Optimized"

    run_name = f"{variation_name_str}_img{current_run_config['TARGET_SIZE'][0]}x{current_run_config['TARGET_SIZE'][1]}_bs{current_run_config['BATCH_SIZE']}_lr{current_run_config['LEARNING_RATE']}"
    run_folder_path = os.path.join('New_runs_mAP50', run_name); os.makedirs(run_folder_path, exist_ok=True) # Distinct folder

    # Setup logging for this specific run
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler) # Remove existing handlers
    log_file_path = os.path.join(run_folder_path, 'training_run.log') # Changed filename
    logging.basicConfig(level=logging.DEBUG if current_run_config.get("DEBUG_LOGGING", False) else logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d - %(message)s',
                        handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)])
    
    logging.info(f"===== Starting Training Run: {run_name} =====")
    logging.info(f"Prioritizing mAP@.50 for model selection. Fallback to F1 if mAP@.50 is zero.")
    logging.info(f"Outputting to folder: {run_folder_path}")
    logging.info(f"Current Run Configuration: {json.dumps(current_run_config, indent=2, sort_keys=True)}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'); logging.info(f"Using device: {device}")

    try:
        train_ds = AbsoluteDataset(images_dir=current_run_config['TRAIN_IMAGE_DIR'], labels_dir=current_run_config['TRAIN_LABEL_DIR'],
                                   target_image_size=current_run_config['TARGET_SIZE'], class_names_for_this_dataset=current_run_config['CLASS_NAMES'])
        val_ds = AbsoluteDataset(images_dir=current_run_config['VAL_IMAGE_DIR'], labels_dir=current_run_config['VAL_LABEL_DIR'],
                                 target_image_size=current_run_config['TARGET_SIZE'], class_names_for_this_dataset=current_run_config['CLASS_NAMES'])
    except FileNotFoundError as e: logging.error(f"Dataset directory not found: {e}. Skipping run."); return
    except Exception as e: logging.error(f"Error initializing dataset: {e}. Skipping run."); return

    if not len(train_ds): logging.error(f"Training dataset is empty ({current_run_config['TRAIN_IMAGE_DIR']}). Skipping run."); return
    if not len(val_ds): logging.warning(f"Validation dataset is empty ({current_run_config['VAL_IMAGE_DIR']}). Validation metrics will be zero or unreliable.")

    train_loader = DataLoader(train_ds, batch_size=current_run_config['BATCH_SIZE'], shuffle=True, collate_fn=collate_fn, num_workers=current_run_config['NUM_WORKERS'], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=current_run_config['BATCH_SIZE'], shuffle=False, collate_fn=collate_fn, num_workers=current_run_config['NUM_WORKERS'], pin_memory=True)
    logging.info(f"Training dataset size: {len(train_ds)}, Validation dataset size: {len(val_ds)}. Number of classes (incl. background): {current_run_config['NUM_CLASSES_MODEL']}")

    model = get_model(num_model_classes=current_run_config['NUM_CLASSES_MODEL']).to(device)
    optimizer = optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=current_run_config['LEARNING_RATE'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=current_run_config.get('SCHEDULER_PATIENCE', 10), verbose=True)
    
    accum_steps = max(1, current_run_config['EFFECTIVE_BATCH'] // current_run_config['BATCH_SIZE'])
    logging.info(f"Physical Batch Size: {current_run_config['BATCH_SIZE']}, Target Effective Batch Size: {current_run_config['EFFECTIVE_BATCH']}, Gradient Accumulation Steps: {accum_steps}")

    best_metric_val = -1.0  # Initialize best_metric_val to a low value
    best_epoch = 0
    epochs_no_improve = 0
    metrics_log_list = []
    best_metric_name_overall = "N/A" # Name of the metric that led to the best model

    for epoch in range(current_run_config['NUM_EPOCHS']):
        model.train()
        epoch_total_loss = 0.0
        optimizer.zero_grad() # Zero gradients at the start of epoch accumulation cycle

        for batch_idx, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            try:
                loss_dict = model(images, targets)
                if not loss_dict or not any(torch.is_tensor(l) for l in loss_dict.values()):
                    logging.warning(f"Epoch {epoch+1}, Batch {batch_idx+1}: Invalid loss_dict returned: {loss_dict}. Skipping batch loss.")
                    continue
                loss = sum(l for l in loss_dict.values() if torch.is_tensor(l)) # Ensure only tensor losses are summed
            except Exception as e:
                logging.error(f"Epoch {epoch+1}, Batch {batch_idx+1}: Error during forward pass or loss calculation: {e}", exc_info=True)
                logging.debug(f"Problematic Targets for Batch {batch_idx+1}: {targets}")
                continue # Skip this batch
            
            scaled_loss = loss / accum_steps
            scaled_loss.backward()
            epoch_total_loss += loss.item()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
        
        avg_epoch_loss = epoch_total_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # Validation Phase
        current_f1, current_map50, current_map50_95 = 0.0, 0.0, 0.0
        val_p, val_r = 0.0, 0.0
        custom_eval_metrics = {'overall': {'f1':0, 'precision':0, 'recall':0}}
        coco_eval_metrics = {}

        if val_loader and len(val_loader) > 0 and len(val_ds) > 0:
            custom_eval_metrics = evaluate_model(model, val_loader, device, current_run_config)
            coco_eval_metrics = generate_coco_eval(model, val_loader, device, current_run_config, run_folder_path)
            current_f1 = custom_eval_metrics['overall']['f1']
            current_map50 = coco_eval_metrics.get("mAP@.50", 0.0)
            current_map50_95 = coco_eval_metrics.get('mAP@.50:95', 0.0)
            val_p = custom_eval_metrics['overall']['precision']
            val_r = custom_eval_metrics['overall']['recall']
        else:
            logging.warning(f"Epoch {epoch+1}: Skipping validation due to empty/missing validation loader or dataset. Metrics will be zero.")

        # --- METRIC FOR SCHEDULER (mAP@.50 -> F1) ---
        metric_for_scheduler = 0.0
        chosen_scheduler_metric_name = "N/A"
        if current_map50 > 0.001: # Prioritize mAP@.50
            metric_for_scheduler = current_map50
            chosen_scheduler_metric_name = "mAP@.50"
        else: # Fallback to F1
            metric_for_scheduler = current_f1
            chosen_scheduler_metric_name = "F1"
        scheduler.step(metric_for_scheduler)
        logging.debug(f"Epoch {epoch+1}: Scheduler stepped with {chosen_scheduler_metric_name}: {metric_for_scheduler:.4f}")

        logging.info(f"[Epoch {epoch+1}/{current_run_config['NUM_EPOCHS']}] TrainLoss: {avg_epoch_loss:.4f} | Val F1: {current_f1:.4f} (P:{val_p:.3f} R:{val_r:.3f}) | mAP@.50:{current_map50:.4f} | mAP@.50-.95:{current_map50_95:.4f} | LR:{optimizer.param_groups[0]['lr']:.2e}")
        
        epoch_log_entry = {'epoch': epoch + 1, 'avg_train_loss': avg_epoch_loss,
                           'val_custom_eval': custom_eval_metrics, 'val_coco_eval': coco_eval_metrics,
                           'learning_rate': optimizer.param_groups[0]['lr']}
        metrics_log_list.append(epoch_log_entry)
        with open(os.path.join(run_folder_path, 'metrics_log_every_epoch.json'), 'w') as f: json.dump(metrics_log_list, f, indent=2)

        # --- METRIC FOR BEST MODEL (mAP@.50 -> F1) ---
        current_best_candidate_metric = 0.0
        chosen_metric_name_for_epoch = "N/A (all metrics zero or invalid)"
        if current_map50 > 0.0: # Prioritize mAP@.50
            current_best_candidate_metric = current_map50
            chosen_metric_name_for_epoch = "mAP@.50"
        elif current_f1 > 0.0: # Fallback to F1
            current_best_candidate_metric = current_f1
            chosen_metric_name_for_epoch = "F1"
        
        if current_best_candidate_metric > best_metric_val:
            best_metric_val = current_best_candidate_metric
            best_epoch = epoch + 1
            epochs_no_improve = 0
            best_metric_name_overall = chosen_metric_name_for_epoch # Store the name of the metric for the current best

            torch.save(model.state_dict(), os.path.join(run_folder_path, 'best_model.pth')) # Reverted to original name
            logging.info(f"<<< Best Model Saved! Epoch: {best_epoch}, Metric ({best_metric_name_overall}): {best_metric_val:.4f}")
            
            summary_data = {
                'best_epoch': best_epoch,
                'best_metric_name': best_metric_name_overall,
                'best_metric_value': best_metric_val,
                'final_custom_eval_at_best': custom_eval_metrics,
                'final_coco_eval_at_best': coco_eval_metrics,
                'config_summary': {k: v for k, v in current_run_config.items() if not (isinstance(v, list) and len(v) > 10) }
            }
            with open(os.path.join(run_folder_path, 'best_metrics_summary.json'), 'w') as f: json.dump(summary_data, f, indent=2) # Reverted
        else:
            epochs_no_improve += 1
            logging.info(f"Epoch {epoch+1}: Metric ({chosen_metric_name_for_epoch}): {current_best_candidate_metric:.4f} did not improve from best ({best_metric_name_overall}): {best_metric_val:.4f}. No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= current_run_config['EARLY_STOPPING_PATIENCE']:
            logging.info(f"===== Early stopping triggered at epoch {epoch+1} as {best_metric_name_overall} did not improve from {best_metric_val:.4f} for {epochs_no_improve} consecutive epochs. =====")
            break
            
    logging.info(f"===== Finished Training Run: {run_name}. Best model from Epoch {best_epoch} with {best_metric_name_overall}: {best_metric_val:.4f}. Log files saved in: {run_folder_path} =====")


if __name__ == '__main__':
    # Ensure the main directory for these runs exists
    os.makedirs('New_runs_mAP50', exist_ok=True) # Updated main runs directory
    
    for i, variation_params in enumerate(TRAINING_VARIATIONS):
        # Ensure VARIATION_NAME is set, or generate a default one
        if "VARIATION_NAME" not in variation_params:
            lr_str = variation_params.get('LEARNING_RATE', 'NA')
            bs_str = variation_params.get('BATCH_SIZE', 'NA')
            sz_tuple = variation_params.get('TARGET_SIZE', ('NA','NA'))
            sz_str = sz_tuple[0] if isinstance(sz_tuple, tuple) and len(sz_tuple) > 0 else 'NA'
            variation_params["VARIATION_NAME"] = f"Run_{i+1}_LR{lr_str}_BS{bs_str}_SZ{sz_str}"
        
        print(f"\n{'='*40}\nProcessing Variation {i+1}/{len(TRAINING_VARIATIONS)}: {variation_params['VARIATION_NAME']}\n{'='*40}")
        train_model_with_variation(i, variation_params)
        print(f"\n{'='*40}\nFinished Variation {i+1}/{len(TRAINING_VARIATIONS)}: {variation_params['VARIATION_NAME']}\n{'='*40}\n")
        
    print("All training variations processed successfully.")