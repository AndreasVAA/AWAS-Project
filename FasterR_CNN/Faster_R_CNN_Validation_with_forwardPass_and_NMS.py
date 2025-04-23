import os
import time
import json
import cv2
import logging
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Collate function to keep batch elements as lists
def collate_fn(batch):
    return tuple(zip(*batch))

# Compute IoU between two bounding boxes
def compute_iou(box1, box2):
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

# Dataset for absolute-pixel labels
def parse_dataset(images_dir, labels_dir):
    all_imgs = []
    for f in os.listdir(images_dir):
        if f.lower().endswith(('.jpg', '.png')):
            lbl = os.path.join(labels_dir, os.path.splitext(f)[0] + '.txt')
            if os.path.exists(lbl) and os.path.getsize(lbl)>0:
                all_imgs.append(f)
    logging.info("Loaded %d samples from %s", len(all_imgs), images_dir)
    return all_imgs

class ValidationDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = parse_dataset(images_dir, labels_dir)
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.images_dir, img_name)
        img = cv2.imread(img_path)
        if img is None: raise RuntimeError(f"Can't load {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img.shape[:2]
        boxes, labels = [], []
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0]+'.txt')
        with open(label_path) as f:
            for line in f:
                parts=line.strip().split()
                if len(parts)!=5:
                    logging.error("Bad label %s: %s", label_path, line.strip())
                    continue
                cls=int(parts[0]); xmin,ymin,xmax,ymax=map(float,parts[1:])
                boxes.append([xmin,ymin,xmax,ymax]); labels.append(cls)
        img_t = torch.from_numpy(img.astype(np.float32)/255.).permute(2,0,1)
        tgt = {"boxes":torch.tensor(boxes, dtype=torch.float32) if boxes else torch.empty((0,4),dtype=torch.float32),
               "labels":torch.tensor(labels, dtype=torch.int64) if labels else torch.empty((0,),dtype=torch.int64),
               "image_id":torch.tensor([idx])}
        return img_t, tgt

# Main validation with timing and metrics
def validate_with_timing(weights_path, images_dir, labels_dir, output_dir,
                         device='cuda', confidence_threshold=0.4, num_classes=2):
    os.makedirs(output_dir,exist_ok=True)
    logging.info("Validation start: %s, images %s, labels %s, output %s", weights_path, images_dir, labels_dir, output_dir)
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    ds = ValidationDataset(images_dir, labels_dir)
    loader = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Model setup
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt.get('model_state_dict', ckpt))
    model.to(device).eval()

    # Warm-up
    dummy = torch.zeros((3, ds[0][0].shape[1], ds[0][0].shape[2]), device=device)
    with torch.no_grad():
        for _ in range(10): model([dummy]); torch.cuda.synchronize()

    # Prepare accumulators
    inference_times = []
    all_ann = []
    all_det = []
    img_info = {}
    ann_id = 1
    per_class = {c:{'tp':0,'fp':0,'fn':0} for c in range(1, num_classes)}

    # Inference loop
    for imgs, tgts in loader:
        img, tgt = imgs[0].to(device), tgts[0]
        # Time forward+NMS
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); start.record()
        with torch.no_grad(): out = model([img])[0]
        end.record(); torch.cuda.synchronize()
        t = start.elapsed_time(end)/1000.0
        inference_times.append(t)

        # Image info
        iid = int(tgt['image_id'].item()); _,H,W = img.shape
        img_info[iid] = {'id':iid,'width':W,'height':H,'file_name':f"{iid}.jpg"}

        # Ground truth and predictions
        gt_boxes = tgt['boxes'].cpu().numpy()
        gt_labels = tgt['labels'].cpu().numpy()
        scores = out['scores'].cpu().numpy()
        keep = scores >= confidence_threshold
        pd_boxes = out['boxes'].cpu().numpy()[keep]
        pd_labels= out['labels'].cpu().numpy()[keep]
        pd_scores= scores[keep]

        # Collect COCO annotations
        for b,l in zip(gt_boxes, gt_labels):
            x1,y1,x2,y2 = b; w,h = x2-x1, y2-y1
            all_ann.append({'id':ann_id,'image_id':iid,'category_id':int(l),'bbox':[x1,y1,w,h],'area':w*h,'iscrowd':0})
            ann_id+=1
        for b,s,l in zip(pd_boxes, pd_scores, pd_labels):
            x1,y1,x2,y2 = b; w,h = x2-x1, y2-y1
            all_det.append({'image_id':iid,'category_id':int(l),'bbox':[x1,y1,w,h],'score':float(s)})

        # Manual per-class TP/FP/FN at IoU>=0.5
        for cls in range(1, num_classes):
            gt_cls = [box for box,lab in zip(gt_boxes,gt_labels) if lab==cls]
            pd_cls = [box for box,lab in zip(pd_boxes,pd_labels) if lab==cls]
            matched = set()
            # True positives & false positives
            for pb in pd_cls:
                found=False
                for j,gb in enumerate(gt_cls):
                    if j in matched: continue
                    if compute_iou(pb,gb)>=0.5:
                        per_class[cls]['tp']+=1; matched.add(j); found=True; break
                if not found: per_class[cls]['fp']+=1
            # False negatives
            per_class[cls]['fn']+= (len(gt_cls)-len(matched))

    # COCO evaluation
    coco_gt = COCO()
    coco_gt.dataset = {'images':list(img_info.values()), 'annotations':all_ann,
                       'categories':[{'id':c,'name':str(c)} for c in range(1,num_classes)]}
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(all_det)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.params.iouThrs = np.linspace(0.5,0.95,10)
    coco_eval.evaluate(); coco_eval.accumulate(); coco_eval.summarize()
    stats = getattr(coco_eval,'stats',[])
    mAP50_95 = float(stats[0]) if len(stats)>0 else 0.0
    mAP50    = float(stats[1]) if len(stats)>1 else 0.0
    mAP75    = float(stats[2]) if len(stats)>2 else 0.0

    # Per-class mAP using COCOeval filtered by category
    _per_c_stats = {}
    for cls in range(1, num_classes):
        coco_eval.params.catIds = [cls]
        coco_eval.evaluate(); coco_eval.accumulate();
        cstats = getattr(coco_eval, 'stats', [])
        _per_c_stats[cls] = [
            float(cstats[0]) if len(cstats)>0 else 0.0,
            float(cstats[1]) if len(cstats)>1 else 0.0,
            float(cstats[2]) if len(cstats)>2 else 0.0
        ]
    mAP50_95 = float(stats[0]) if len(stats)>0 else 0.0
    mAP50    = float(stats[1]) if len(stats)>1 else 0.0
    mAP75    = float(stats[2]) if len(stats)>2 else 0.0

    # Manual global precision/recall/F1
    total_tp = sum(per_class[c]['tp'] for c in per_class)
    total_fp = sum(per_class[c]['fp'] for c in per_class)
    total_fn = sum(per_class[c]['fn'] for c in per_class)
    precision = total_tp/(total_tp+total_fp) if total_tp+total_fp>0 else 0.0
    recall    = total_tp/(total_tp+total_fn) if total_tp+total_fn>0 else 0.0
    f1        = 2*precision*recall/(precision+recall) if precision+recall>0 else 0.0

    # Timing summary
    total_time = sum(inference_times)
    avg_time = (total_time/len(inference_times))*1000.0 if inference_times else 0.0
    fps = len(ds)/total_time if total_time>0 else 0.0

    # Build output JSON
    summary = {
        'Precision': precision,
        'Recall': recall,
        'F1': f1,
        'mAP50': mAP50,
        'mAP50-95': mAP50_95,
        'mAP75': mAP75,
        'Confidence Threshold': confidence_threshold,
        'Core GPU Time (s)': total_time,
        'Core Avg Time per Image (ms)': avg_time,
        'Core FPS': fps,
        'Num Images': len(ds),
        'Per Class': {
        # include per-class precision/recall/F1/TP/FP/FN/support and per-class mAP50, mAP50-95, mAP75
        **{str(c): {
            'precision': (per_class[c]['tp'] / (per_class[c]['tp'] + per_class[c]['fp']) if (per_class[c]['tp'] + per_class[c]['fp']) > 0 else 0.0),
            'recall':    (per_class[c]['tp'] / (per_class[c]['tp'] + per_class[c]['fn']) if (per_class[c]['tp'] + per_class[c]['fn']) > 0 else 0.0),
            'f1':        (2 * per_class[c]['tp'] / (2 * per_class[c]['tp'] + per_class[c]['fp'] + per_class[c]['fn']) if (per_class[c]['tp'] + per_class[c]['fp'] + per_class[c]['fn']) > 0 else 0.0),
            'tp': per_class[c]['tp'],
            'fp': per_class[c]['fp'],
            'fn': per_class[c]['fn'],
            'support': per_class[c]['tp'] + per_class[c]['fn'],
            # per-class mAP via fresh COCOeval
            **({
                'mAP50-95': float(_per_c_stats[c][0]),
                'mAP50'   : float(_per_c_stats[c][1]),
                'mAP75'   : float(_per_c_stats[c][2])
            } if c in _per_c_stats else {'mAP50-95': 0.0, 'mAP50': 0.0, 'mAP75': 0.0})
        } for c in per_class}
    }
    }

    with open(os.path.join(output_dir,'validation_metrics.json'),'w') as f:
        json.dump(summary, f, indent=2)
    logging.info("Saved metrics to %s", os.path.join(output_dir,'validation_metrics.json'))
    return summary

if __name__=='__main__':
    validate_with_timing(
        weights_path='/home/itk/Desktop/Andreas/AWAS-Project/FasterR_CNN/runs/Testing_new_learning_rate_set_to_0.01/best_model_absolute.pth',
        images_dir='/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/images',
        labels_dir='/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels_minmax',
        output_dir='Validation_FasterRCNN__lr_set_0.01_with_forwardPass_and_NMS',
        device='cuda',
        confidence_threshold=0.4,
        num_classes=2
    )
    print("Validation completed. Results saved in the output directory.")