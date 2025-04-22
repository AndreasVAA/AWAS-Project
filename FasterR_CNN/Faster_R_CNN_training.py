import os                     # For file and directory operations
import cv2                    # OpenCV for image loading and processing
import torch                  # PyTorch library for deep learning
import numpy as np            # For numerical operations
import torchvision            # For pre-trained models and vision utilities
import torch.optim as optim   # For optimization algorithms (SGD, etc.)
from torch.utils.data import Dataset, DataLoader, Subset  # Data handling and batching
import torchvision.transforms.functional as TF  # Image transformation utilities
import time                   # For measuring training time
import logging
import json
import random
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights  # Pre-trained weights

# Import augmentation functions from augmentations.py (assumed to be in same folder)
from augmentations import get_train_transforms, mixup_augment, get_randaugment_pipeline, mosaic_augment

#############################################
# SINGLE PLACE TO CHANGE IMAGE SIZE
#############################################
TARGET_SIZE = (1280, 960)  # (width, height)

#############################################
# Setup Run Folder and Logging
#############################################
runs_dir = os.path.join(os.getcwd(), "runs")
if not os.path.exists(runs_dir):
    os.makedirs(runs_dir)

run_name = "Testing_new_learning_rate_set_to_0.01"  # Change as desired for each run
run_folder = os.path.join(runs_dir, run_name)
os.makedirs(run_folder, exist_ok=True)

log_file = os.path.join(run_folder, "training.log")
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename=log_file,
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

#############################################
# Dataset Definitions
#############################################
class AbsoluteDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
        valid_images = []
        for img in all_images:
            label_path = os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                if content:
                    valid_images.append(img)
                else:
                    logging.warning("Empty label file for image '%s'", img)
            else:
                logging.warning("No label file for image '%s'", img)
        self.images = valid_images
        logging.info("[AbsoluteDataset] Initialized with %d images.", len(self.images))
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_file = os.path.join(self.labels_dir, os.path.splitext(img_file)[0] + ".txt")
        boxes = []
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    logging.warning("Skipping malformed line in %s: %s", label_file, line)
                    continue
                class_id = int(parts[0])
                xmin, ymin, xmax, ymax = map(float, parts[1:])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0,4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels, "image_id": torch.tensor([idx])}
        if boxes.numel() > 0:
            area = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        else:
            area = torch.empty((0,), dtype=torch.float32)
        target["area"] = area
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        img = TF.to_tensor(img)
        return img, target

# === Modified AugmentedDataset ===
class AugmentedDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform_yolo, transform_randaug, 
                 mosaic_prob=1.0, mixup_prob=0.2, mixup_alpha=32):
        """
        mosaic_prob: probability to apply mosaic augmentation.
        (mixup_prob and randaugment remain available and are commented out in __getitem__.)
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform_yolo = transform_yolo
        self.transform_randaug = transform_randaug
        self.mosaic_prob = mosaic_prob
        self.mixup_prob = mixup_prob
        self.mixup_alpha = mixup_alpha

        all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
        valid_images = []
        for img in all_images:
            label_path = os.path.join(labels_dir, os.path.splitext(img)[0] + ".txt")
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    content = f.read().strip()
                if content:
                    valid_images.append(img)
                else:
                    print(f"Warning: Empty label file for image '{img}'")
            else:
                print(f"Warning: No label file for image '{img}'")
        self.images = valid_images

    def __len__(self):
        return len(self.images)

    def load_image_and_annotations(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_file = os.path.join(self.labels_dir, os.path.splitext(img_file)[0] + ".txt")
        boxes = []
        labels = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(parts[0])
                xmin, ymin, xmax, ymax = map(float, parts[1:])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(cls_id)
        return img, boxes, labels

    def __getitem__(self, idx):
        # Decide whether to use mosaic augmentation based on mosaic_prob.
        if random.random() < self.mosaic_prob:
            # Mosaic: sample 4 images (if available)
            indices = [idx] + random.sample([i for i in range(len(self.images)) if i != idx],
                                             min(3, len(self.images)-1))
            while len(indices) < 4:
                indices.append(indices[-1])
            mosaic_imgs = []
            mosaic_boxes_list = []
            mosaic_labels_list = []
            for i in indices:
                im, b, l = self.load_image_and_annotations(i)
                mosaic_imgs.append(im)
                mosaic_boxes_list.append(b)
                mosaic_labels_list.append(l)
            # Apply mosaic augmentation.
            mosaic_img, mosaic_boxes, mosaic_labels = mosaic_augment(
                mosaic_imgs, mosaic_boxes_list, mosaic_labels_list, target_size=TARGET_SIZE)
            data = {"image": mosaic_img, "bboxes": mosaic_boxes, "category_ids": mosaic_labels}
        else:
            # Otherwise, load just a single image.
            im, b, l = self.load_image_and_annotations(idx)
            data = {"image": im, "bboxes": b, "category_ids": l}

        # Apply YOLO-style augmentation to the image (whether mosaic or single).
        augmented = self.transform_yolo(**data)
        img_yolo = augmented["image"]
        boxes_yolo = augmented["bboxes"]
        labels_yolo = augmented["category_ids"]

        # The following blocks for RandAugment and mixup are left commented out.
        #"""
        # RandAugment (if desired)
        data2 = {"image": img_yolo, "bboxes": boxes_yolo, "category_ids": labels_yolo}
        augmented_rand = self.transform_randaug(**data2)
        img_rand = augmented_rand["image"]
        boxes_rand = augmented_rand["bboxes"]
        labels_rand = augmented_rand["category_ids"]
        final_img = img_rand
        final_boxes = boxes_rand
        final_labels = labels_rand
        #"""
        # In our current run, we simply use the mosaic/YOLO output.
        final_img = img_yolo
        final_boxes = boxes_yolo
        final_labels = labels_yolo

        # Convert final image to tensor.
        final_img = final_img.astype(np.float32) / 255.0
        final_img_tensor = torch.from_numpy(final_img).permute(2, 0, 1)

        target = {
            "boxes": torch.tensor(final_boxes, dtype=torch.float32) if final_boxes else torch.empty((0, 4), dtype=torch.float32),
            "labels": torch.tensor(final_labels, dtype=torch.int64) if final_labels else torch.empty((0,), dtype=torch.int64),
            "image_id": torch.tensor([idx])
        }
        if target["boxes"].numel() > 0:
            area = (target["boxes"][:, 2] - target["boxes"][:, 0]) * (target["boxes"][:, 3] - target["boxes"][:, 1])
        else:
            area = torch.empty((0,), dtype=torch.float32)
        target["area"] = area
        target["iscrowd"] = torch.zeros((target["boxes"].shape[0],), dtype=torch.int64)

        return final_img_tensor, target

# Custom collate function for batching.
def collate_fn(batch):
    return tuple(zip(*batch))

#############################################
# Model Setup
#############################################
def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    return model

#############################################
# Helper: Compute IoU between two boxes
#############################################
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area

#############################################
# Helper: Evaluate Model on Validation Set
#############################################
def evaluate_model(model, data_loader, device, score_threshold=0.3, iou_threshold=0.3):
    total_tp = total_fp = total_fn = total_images = 0
    inference_times = []
    per_image_times = []
    all_annotations = []
    all_detections = []
    image_infos = {}
    annotation_id = 1

    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            start_inference = time.time()
            outputs = model(images)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            batch_time = time.time() - start_inference
            inference_times.append(batch_time)
            batch_size = len(images)
            per_image_times.extend([batch_time / batch_size] * batch_size)
            
            for i, (img, target, output) in enumerate(zip(images, targets, outputs)):
                total_images += 1
                gt_boxes = target["boxes"].cpu().numpy()
                num_gt = len(gt_boxes)
                pred_scores = output["scores"].cpu().numpy()
                valid_idx = pred_scores >= score_threshold
                filtered_boxes = output["boxes"].cpu().numpy()[valid_idx]
                num_filtered = len(filtered_boxes)
                
                tp = 0
                matched = set()
                for pred_box in filtered_boxes:
                    for j, gt_box in enumerate(gt_boxes):
                        if j in matched:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou >= iou_threshold:
                            tp += 1
                            matched.add(j)
                            break
                fp = num_filtered - tp
                fn = num_gt - tp
                total_tp += tp
                total_fp += fp
                total_fn += fn

                image_id = target["image_id"].item()
                if image_id not in image_infos:
                    _, H, W = img.shape
                    image_infos[image_id] = {
                        "id": image_id,
                        "width": W,
                        "height": H,
                        "file_name": f"image_{image_id}.jpg"
                    }
                for box in gt_boxes:
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    all_annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [float(box[0]), float(box[1]), float(w), float(h)],
                        "area": float(w * h),
                        "iscrowd": 0
                    })
                    annotation_id += 1
                pred_boxes = output["boxes"].cpu().numpy()[valid_idx]
                pred_scores = output["scores"].cpu().numpy()[valid_idx]
                for box, score in zip(pred_boxes, pred_scores):
                    w = box[2] - box[0]
                    h = box[3] - box[1]
                    all_detections.append({
                        "image_id": image_id,
                        "category_id": 1,
                        "bbox": [float(box[0]), float(box[1]), float(w), float(h)],
                        "score": float(score)
                    })

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    total_inference_time = sum(inference_times)
    avg_inference_time = sum(per_image_times) / len(per_image_times) if per_image_times else 0.0
    max_inference_time = max(per_image_times) if per_image_times else 0.0
    fps = total_images / total_inference_time if total_inference_time > 0 else 0.0

    logging.info("[Eval] Overall Precision: %.4f", precision)
    logging.info("[Eval] Overall Recall:    %.4f", recall)
    logging.info("[Eval] Overall F1 Score:  %.4f", f1)
    logging.info("[Eval] Inference - Total time: %.4fs, Avg time per image: %.4fs, Max time per image: %.4fs, FPS: %.2f",
                 total_inference_time, avg_inference_time, max_inference_time, fps)

    eval_metrics = {
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "inference": {
            "total_time": total_inference_time,
            "avg_time": avg_inference_time,
            "max_time": max_inference_time,
            "fps": fps
        }
    }
    
    # -- Compute mAP metrics using pycocotools --
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        coco_gt_dict = {
            "images": list(image_infos.values()),
            "annotations": all_annotations,
            "categories": [{"id": 1, "name": "object"}]
        }
        gt_path = os.path.join(run_folder, "temp_coco_gt.json")
        with open(gt_path, "w") as f:
            json.dump(coco_gt_dict, f)
        
        cocoGt = COCO(gt_path)
        
        dets_path = os.path.join(run_folder, "temp_coco_dets.json")
        with open(dets_path, "w") as f:
            json.dump(all_detections, f)
        
        cocoDt = cocoGt.loadRes(dets_path)
        cocoEval = COCOeval(cocoGt, cocoDt, iouType='bbox')
        # Set multiple IoU thresholds for mAP50-95 (default in COCO evaluation)
        cocoEval.params.iouThrs = np.linspace(0.5, 0.95, 10)
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        mAP50 = cocoEval.stats[1]  # sometimes index 1 is used for mAP50
        mAP50_95 = cocoEval.stats[0]  # mean AP over IoU thresholds
        eval_metrics["coco"] = {"mAP50": mAP50, "mAP50-95": mAP50_95}
        logging.info("[Eval] COCO mAP50: %.4f", mAP50)
        logging.info("[Eval] COCO mAP50-95: %.4f", mAP50_95)
        os.remove(gt_path)
        os.remove(dets_path)
    except Exception as e:
        logging.warning("COCO evaluation skipped: " + str(e))
        eval_metrics["coco"] = {"mAP50": 0.0, "mAP50-95": 0.0}

    return eval_metrics


#############################################
# Modified Training Function: Detailed Loss Tracking
#############################################
def train_model(train_loader, val_loader, model, device, num_epochs=500, learning_rate=0.01,
                score_threshold=0.3, iou_threshold=0.3):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.1, patience=10, verbose=True)

    best_f1 = 0.0
    best_epoch = 0
    best_eval_metrics = None
    epochs_without_improvement = 0
    early_stop_patience = 50
    target_decay_epoch = 20
    warmup_epochs = 3

    # Store initial mosaic probability if available.
    if hasattr(train_loader.dataset, 'mosaic_prob'):
        if not hasattr(train_loader.dataset, 'initial_mosaic_prob'):
            train_loader.dataset.initial_mosaic_prob = train_loader.dataset.mosaic_prob

    # For loss breakdown, we accumulate each loss component over the epoch.
    for epoch in range(num_epochs):
        if epoch < warmup_epochs:
            warmup_lr = learning_rate * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr
            logging.info("Epoch %d: Warmup learning rate set to %.6f", epoch+1, warmup_lr)

        if hasattr(train_loader.dataset, 'mosaic_prob'):
            init_prob = train_loader.dataset.initial_mosaic_prob
            decay_factor = max(1 - (epoch / target_decay_epoch) ** 2, 0.1)
            new_prob = init_prob * decay_factor
            train_loader.dataset.mosaic_prob = new_prob
            logging.info("Epoch %d: Updated mosaic probability to %.4f", epoch+1, new_prob)

        model.train()
        epoch_loss = 0.0
        # Dictionary to accumulate individual loss components.
        epoch_loss_components = {}
        start_time = time.time()
        logging.info("[Training] Epoch %d/%d started...", epoch+1, num_epochs)

        for batch_idx, (images, targets) in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                print("DEBUG: First sample target in training batch:", targets[0])
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

            loss_dict = model(images, targets)
            if isinstance(loss_dict, list):
                logging.error("[Training] Error: Model returned predictions during training.")
                continue

            # Log each component (e.g., "loss_classifier", "loss_box_reg", etc.)
            for key, loss_val in loss_dict.items():
                epoch_loss_components.setdefault(key, 0.0)
                epoch_loss_components[key] += loss_val.item()

            batch_loss = sum(loss for loss in loss_dict.values())
            epoch_loss += batch_loss.item()

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

        avg_loss = epoch_loss / len(train_loader)
        elapsed = time.time() - start_time
        current_lr = optimizer.param_groups[0]['lr']
        logging.info("[Training] Epoch %d finished: Average Total Loss = %.4f, Time = %.2fs, LR = %.6f",
                     epoch+1, avg_loss, elapsed, current_lr)
        # Log average loss per component.
        for key, total_val in epoch_loss_components.items():
            avg_component = total_val / len(train_loader)
            logging.info("[Training] Epoch %d: Average %s = %.4f", epoch+1, key, avg_component)

        logging.info("[Validation] Evaluating model on validation set...")
        current_eval = evaluate_model(model, val_loader, device, score_threshold, iou_threshold)
        current_f1 = current_eval["f1"]

        lr_scheduler.step(current_f1)

        if current_f1 > best_f1:
            best_f1 = current_f1
            best_epoch = epoch + 1
            best_eval_metrics = current_eval
            best_model_path = os.path.join(run_folder, "best_model_absolute.pth")
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'f1': current_f1,
            }, best_model_path)
            logging.info("--> [Checkpoint] Best model updated at epoch %d with F1 = %.4f", best_epoch, best_f1)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            logging.info("No improvement for %d epochs", epochs_without_improvement)
            if epochs_without_improvement >= early_stop_patience:
                logging.info("Early stopping triggered after %d epochs with no improvement", early_stop_patience)
                break

    logging.info("[Training] Complete. Best epoch: %d with F1 Score = %.4f", best_epoch, best_f1)
    
    # Save a detailed summary.
    metrics_summary_path = os.path.join(run_folder, "metrics_summary.txt")
    with open(metrics_summary_path, "w") as f:
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"Best F1: {best_f1:.4f}\n")
        if best_eval_metrics:
            precision = best_eval_metrics.get("precision", 0.0)
            recall = best_eval_metrics.get("recall", 0.0)
            f.write(f"Overall Precision: {precision:.4f}\n")
            f.write(f"Overall Recall:    {recall:.4f}\n")
        else:
            f.write("Overall Precision: N/A\n")
            f.write("Overall Recall:    N/A\n")
        # Log training hyperparameters.
        f.write("Hyperparameters:\n")
        f.write(f"  Batch Size: {train_loader.batch_size}\n")
        f.write(f"  Initial Learning Rate: {learning_rate}\n")
        f.write(f"  Num Epochs: {num_epochs}\n")
        f.write(f"  Score Threshold: {score_threshold}\n")
        f.write(f"  IoU Threshold: {iou_threshold}\n")
        if best_eval_metrics and "coco" in best_eval_metrics:
            coco_mAP50 = best_eval_metrics["coco"].get("mAP50", "N/A")
            coco_mAP50_95 = best_eval_metrics["coco"].get("mAP50-95", "N/A")
            f.write(f"  COCO mAP50: {coco_mAP50}\n")
            f.write(f"  COCO mAP50-95: {coco_mAP50_95}\n")
        else:
            f.write("  COCO mAP: N/A (pycocotools not installed or evaluation skipped)\n")

        # Optionally, add the average loss breakdown per component if you logged it.
        f.write("\nLoss Breakdown per Epoch (averaged over batches):\n")
        for key, total_val in epoch_loss_components.items():
            avg_component = total_val / len(train_loader)
            f.write(f"  {key}: {avg_component:.4f}\n")

    # Save system summary (memory, GPU, etc.) as before.
    system_summary_path = os.path.join(run_folder, "system_summary.txt")
    with open(system_summary_path, "w") as f:
        if best_eval_metrics:
            inf = best_eval_metrics["inference"]
            f.write("Inference Metrics:\n")
            f.write(f"  Total Inference Time: {inf['total_time']:.4f} s\n")
            f.write(f"  Average Inference Time per Image: {inf['avg_time']:.4f} s\n")
            f.write(f"  Max Inference Time per Image: {inf['max_time']:.4f} s\n")
            f.write(f"  FPS: {inf['fps']:.2f}\n")
        else:
            f.write("Inference Metrics: N/A\n")
        if device.type == 'cuda':
            max_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            f.write(f"\nGPU Memory Usage:\n")
            f.write(f"  Max GPU Memory Allocated: {max_mem:.2f} MB\n")
            props = torch.cuda.get_device_properties(device)
            f.write(f"  GPU Name: {props.name}\n")
            f.write(f"  GPU Total Memory: {props.total_memory / (1024 * 1024):.2f} MB\n")
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            vm = psutil.virtual_memory()
            f.write(f"\nSystem Memory:\n")
            f.write(f"  Total System Memory: {vm.total / (1024 * 1024):.2f} MB\n")
            f.write(f"  Available System Memory: {vm.available / (1024 * 1024):.2f} MB\n")
            f.write(f"  CPU Memory Usage (RSS): {mem_info.rss / (1024 * 1024):.2f} MB\n")
            f.write(f"  CPU Count: {psutil.cpu_count(logical=True)}\n")
        except ImportError:
            f.write("psutil not installed; additional system memory info not available.\n")

    return model, best_epoch


#############################################
# Main: Running the Training Pipeline
#############################################
if __name__ == '__main__':
    train_images_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images"
    train_labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/labels_minmax"
    val_images_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/images"
    val_labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels_minmax"

    logging.info("[Main] Initializing datasets...")

    
    # === Training with Augmentation ===
    
    train_dataset = AugmentedDataset(
        train_images_dir, 
        train_labels_dir, 
        transform_yolo=get_train_transforms(), 
        transform_randaug=get_randaugment_pipeline(), 
        mosaic_prob=1.0,    # Starting mosaic probability (e.g. 100%)
        mixup_prob=0.0, 
        mixup_alpha=32
    )
    

    # === Training without Augmentation ===
    # To disable augmentation, you could use AbsoluteDataset:
    #train_dataset = AbsoluteDataset(train_images_dir, train_labels_dir)
    
    val_dataset = AbsoluteDataset(val_images_dir, val_labels_dir)

    batch_size = 6
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    num_classes = 2  # One object class + background
    model = get_model(num_classes)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    logging.info("[Main] Starting training with detection metric-based model selection...")
    trained_model, best_epoch = train_model(
        train_loader, val_loader, model, device, num_epochs=300,
        learning_rate=0.01, score_threshold=0.4, iou_threshold=0.6
    )
