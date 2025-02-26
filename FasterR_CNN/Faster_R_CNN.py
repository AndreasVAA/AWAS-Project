import os                     # For file and directory operations
import cv2                    # OpenCV for image loading and processing
import torch                  # PyTorch library for deep learning
import numpy as np            # For numerical operations
import torchvision            # For pre-trained models and vision utilities
import torch.optim as optim   # For optimization algorithms (SGD, etc.)
from torch.utils.data import Dataset, DataLoader, Subset  # Data handling and batching
import torchvision.transforms.functional as TF  # Image transformation utilities
import time                   # For measuring training time
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights  # Pre-trained weights
import logging

# Set up logging (this will log to both console and a file 'training.log')
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    filename='training.log',
                    filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

#############################################
# Dataset Definition: AbsoluteDataset
#############################################
class AbsoluteDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transforms=None):
        """
        Initializes the dataset for loading images along with their corresponding 
        absolute coordinate labels (bounding boxes). Each label file is expected 
        to contain lines in the format: "class_id xmin ymin xmax ymax".
        """
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transforms = transforms

        # List all image files with jpg or png extension
        all_images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
        valid_images = []
        
        # Iterate over each image to check if it has a corresponding label file
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
        logging.info("[Dataset] Initialized with %d images having valid absolute labels.", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            raise RuntimeError(f"[Dataset] Failed to load image: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        label_file = os.path.join(self.labels_dir, os.path.splitext(img_file)[0] + ".txt")
        boxes = []
        labels = []
        
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    logging.warning("[Dataset] Skipping malformed line in %s: %s", label_file, line)
                    continue
                class_id = int(parts[0])
                xmin, ymin, xmax, ymax = map(float, parts[1:])
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(class_id)
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.empty((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }
        if boxes.numel() > 0:
            area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        else:
            area = torch.empty((0,), dtype=torch.float32)
        target["area"] = area
        target["iscrowd"] = torch.zeros((boxes.shape[0],), dtype=torch.int64)

        if self.transforms:
            img = self.transforms(img)
        else:
            img = TF.to_tensor(img)
        
        return img, target

# Custom collate function
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
def evaluate_model(model, data_loader, device, score_threshold=0.3, iou_threshold=0.3):
    total_tp = total_fp = total_fn = total_images = 0
    model.eval()
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for output, target in zip(outputs, targets):
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
                    for i, gt_box in enumerate(gt_boxes):
                        if i in matched:
                            continue
                        iou = compute_iou(pred_box, gt_box)
                        if iou >= iou_threshold:
                            tp += 1
                            matched.add(i)
                            break
                fp = num_filtered - tp
                fn = num_gt - tp

                total_tp += tp
                total_fp += fp
                total_fn += fn

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    logging.info("[Eval] Evaluation complete!")
    logging.info("[Eval] Total images processed: %d", total_images)
    logging.info("[Eval] Aggregate metrics: TP=%d, FP=%d, FN=%d", total_tp, total_fp, total_fn)
    logging.info("[Eval] Overall Precision: %.4f", precision)
    logging.info("[Eval] Overall Recall:    %.4f", recall)
    logging.info("[Eval] Overall F1 Score:  %.4f", f1)

    return f1

#############################################
# Training Module with Detection Metric-Based Checkpointing
#############################################
def train_model(train_loader, val_loader, model, device, num_epochs=10, learning_rate=0.005,
                score_threshold=0.3, iou_threshold=0.3):
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    
    # Use ReduceLROnPlateau to reduce LR when F1 score plateaus.
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                              mode='max',
                                                              factor=0.1,
                                                              patience=5,
                                                              verbose=True)

    best_f1 = 0.0
    best_epoch = 0

    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()
            logging.info("[Training] Epoch %d/%d started...", epoch+1, num_epochs)

            for batch_idx, (images, targets) in enumerate(train_loader):
                for i, target in enumerate(targets):
                    if target["boxes"].numel() == 0:
                        logging.warning("[Training] Sample %d in batch %d has no bounding boxes!", i, batch_idx)
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in target.items()} for target in targets]

                loss_dict = model(images, targets)
                if isinstance(loss_dict, list):
                    logging.error("[Training] Error: Model returned predictions during training.")
                    continue
                losses = sum(loss for loss in loss_dict.values())
                epoch_loss += losses.item()

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

            avg_loss = epoch_loss / len(train_loader)
            elapsed = time.time() - start_time
            current_lr = optimizer.param_groups[0]['lr']
            logging.info("[Training] Epoch %d finished: Average Loss = %.4f, Time = %.2fs, LR = %.6f",
                         epoch+1, avg_loss, elapsed, current_lr)

            logging.info("[Validation] Evaluating model on validation set...")
            current_f1 = evaluate_model(model, val_loader, device, score_threshold, iou_threshold)

            # Step the scheduler with the current F1 score
            lr_scheduler.step(current_f1)

            if current_f1 > best_f1:
                best_f1 = current_f1
                best_epoch = epoch + 1
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1': current_f1,
                }, "best_model_absolute.pth")
                logging.info("--> [Checkpoint] Best model updated at epoch %d with F1 = %.4f", best_epoch, best_f1)

            # Optionally, save periodic checkpoints (e.g., every 10 epochs)
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}_f1_{current_f1:.4f}.pth"
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'f1': current_f1,
                }, checkpoint_path)
                logging.info("Checkpoint saved: %s", checkpoint_path)

    except KeyboardInterrupt:
        logging.info("Training interrupted. Saving current model state.")
        torch.save(model.state_dict(), "model_interrupt.pth")
        raise

    logging.info("[Training] Complete. Best epoch: %d with F1 Score = %.4f", best_epoch, best_f1)
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
    train_dataset = AbsoluteDataset(train_images_dir, train_labels_dir)
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
        train_loader, val_loader, model, device, num_epochs=500,
        learning_rate=0.005, score_threshold=0.3, iou_threshold=0.3
    )
