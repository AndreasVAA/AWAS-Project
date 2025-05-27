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
from augmentations import get_train_transforms, get_randaugment_pipeline, mosaic_augment
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# === CONFIGURATION ===
CONFIG = {
    "TARGET_SIZE": (640, 640),
    "BATCH_SIZE": 4,
    "EFFECTIVE_BATCH": 64,
    "NUM_CLASSES": 6,
    "NUM_EPOCHS": 400,
    "LEARNING_RATE": 0.01,
    "SCORE_THRESHOLD": 0.4,
    "IOU_THRESHOLD": 0.6,
    "EARLY_STOPPING_PATIENCE": 100,
    "TRAIN_IMAGE_DIR": "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/images",
    "TRAIN_LABEL_DIR": "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/labels_minmax",
    "VAL_IMAGE_DIR":   "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/val/images",
    "VAL_LABEL_DIR":   "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/val/labels_minmax",
    "CLASS_NAMES": ["Tripos longipes", "Tripos fusus", "Tripos lineatum/furca", "Chaetoceros", "Coscinodiscus"],

}


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_iou(box1, box2):
    # box = [x1, y1, x2, y2]
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter/union if union>0 else 0
class AbsoluteDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx]
        img_path = os.path.join(self.images_dir, img_file)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_file)[0] + '.txt')
        boxes, labels = [], []
        with open(label_path) as f:
            for line in f:
                cls, x, y, w, h = map(float, line.strip().split())
                x1, y1 = x - w/2, y - h/2
                x2, y2 = x + w/2, y + h/2
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls))
        img_tensor = TF.to_tensor(img)
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([idx], dtype=torch.int64)
        }
        return img, img_tensor, target


def collate_fn(batch):
    return tuple(zip(*batch))

def get_model(num_classes):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_feats = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_feats, num_classes)
    return model


def evaluate_model(model, data_loader, device, score_thresh=CONFIG['SCORE_THRESHOLD'], iou_thresh=CONFIG['IOU_THRESHOLD']):
    # per-class counters
    tp = [0]*CONFIG['NUM_CLASSES']; fp=[0]*CONFIG['NUM_CLASSES']; fn=[0]*CONFIG['NUM_CLASSES']
    model.eval(); inf_times=[]; perimg=[]
    with torch.no_grad():
        for imgs, tgts in data_loader:
            imgs = [i.to(device) for i in imgs]
            start=time.time(); outs=model(imgs)
            if device.type=='cuda': torch.cuda.synchronize()
            dt=time.time()-start; inf_times.append(dt); perimg+=[dt/len(imgs)]*len(imgs)
            for out, tgt in zip(outs, tgts):
                gt_boxes = tgt['boxes'].cpu().numpy(); gt_lbls=tgt['labels'].cpu().numpy()
                pr_sc, pr_bx, pr_lbl = out['scores'].cpu().numpy(), out['boxes'].cpu().numpy(), out['labels'].cpu().numpy()
                keep = pr_sc>=score_thresh
                pr_bx, pr_lbl = pr_bx[keep], pr_lbl[keep]
                matched = set()
                for pb, pl in zip(pr_bx, pr_lbl):
                    c = pl
                    best_iou=0; best_j=-1
                    for j, (gb, gl) in enumerate(zip(gt_boxes, gt_lbls)):
                        if gl!=c or j in matched: continue
                        iou=compute_iou(pb,gb)
                        if iou>best_iou: best_iou, best_j = iou,j
                    if best_iou>=iou_thresh:
                        tp[c]+=1; matched.add(best_j)
                    else: fp[c]+=1
                # false negatives
                for j, gl in enumerate(gt_lbls):
                    if j not in matched: fn[gl]+=1
    # compute metrics
    cls_metrics={}; o_tp,o_fp,o_fn=0,0,0
    for i,name in enumerate(CONFIG['CLASS_NAMES']):
        P = tp[i]/(tp[i]+fp[i]) if tp[i]+fp[i]>0 else 0
        R = tp[i]/(tp[i]+fn[i]) if tp[i]+fn[i]>0 else 0
        F = 2*P*R/(P+R) if P+R>0 else 0
        cls_metrics[name]={'precision':P,'recall':R,'f1':F}
        o_tp+=tp[i]; o_fp+=fp[i]; o_fn+=fn[i]
    P = o_tp/(o_tp+o_fp) if o_tp+o_fp>0 else 0
    R = o_tp/(o_tp+o_fn) if o_tp+o_fn>0 else 0
    F = 2*P*R/(P+R) if P+R>0 else 0
    inf_total=sum(inf_times);inf_avg=np.mean(perimg) if perimg else 0;inf_max=max(perimg) if perimg else 0; fps=len(perimg)/inf_total if inf_total>0 else 0
    overall={'precision':P,'recall':R,'f1':F,'inference':{'total_time':inf_total,'avg_time':inf_avg,'max_time':inf_max,'fps':fps}}
    return {'overall':overall,'per_class':cls_metrics}



def generate_coco_eval(model, data_loader, device, class_names, run_folder):
    model.eval()
    coco_gt = {"images": [], "annotations": [], "categories": []}
    coco_dt = []

    ann_id = 1
    image_id = 0
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for img_tensor, output, target in zip(images, outputs, targets):
                coco_gt["images"].append({
                    "id": int(image_id),
                    "width": img_tensor.shape[2],
                    "height": img_tensor.shape[1]
                })
                boxes = target['boxes'].cpu().numpy()
                labels = target['labels'].cpu().numpy()
                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = box
                    coco_gt["annotations"].append({
                        "id": ann_id,
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "area": float((x2 - x1) * (y2 - y1)),
                        "iscrowd": 0
                    })
                    ann_id += 1

                for box, score, label in zip(output["boxes"].cpu().numpy(),
                                              output["scores"].cpu().numpy(),
                                              output["labels"].cpu().numpy()):
                    x1, y1, x2, y2 = box
                    coco_dt.append({
                        "image_id": int(image_id),
                        "category_id": int(label),
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": float(score)
                    })
                image_id += 1

    for i, name in enumerate(class_names):
        coco_gt["categories"].append({"id": i, "name": name})

    coco_gt_obj = COCO()
    coco_gt_obj.dataset = coco_gt
    coco_gt_obj.createIndex()
    coco_dt_obj = coco_gt_obj.loadRes(coco_dt)

    coco_eval = COCOeval(coco_gt_obj, coco_dt_obj, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    metrics = {
        "mAP@.50": stats[1],
        "mAP@.75": stats[2],
        "mAP@.50:95": stats[0],
        "AR@1": stats[5],
        "AR@10": stats[6]
    }

    # Per-class AP
    per_class_ap = {}
    precisions = coco_eval.eval["precision"]  # [T, R, K, A, M]
    for i, name in enumerate(class_names):
        ap = precisions[0, :, i, 0, 0]
        valid = ap[ap > -1]
        per_class_ap[name] = float(np.mean(valid)) if len(valid) else 0.0

    metrics["per_class_ap"] = per_class_ap

    # Save per-class AP
    with open(os.path.join(run_folder, "per_class_ap.json"), "w") as f:
        json.dump(per_class_ap, f, indent=2)

    return metrics


if __name__=='__main__':
    seed_everything()
    CONFIG['accum_steps']=CONFIG['EFFECTIVE_BATCH']//CONFIG['BATCH_SIZE']
    CONFIG['run_name']=f"RCNN_{CONFIG['TARGET_SIZE'][0]}x{CONFIG['TARGET_SIZE'][1]}_bs{CONFIG['BATCH_SIZE']}_acc{CONFIG['EFFECTIVE_BATCH']}"
    CONFIG['run_folder']=os.path.join('runs',CONFIG['run_name']); os.makedirs(CONFIG['run_folder'],exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(CONFIG['run_folder'],'train.log')), logging.StreamHandler()])
    # datasets & loaders
    train_ds=AbsoluteDataset(CONFIG['TRAIN_IMAGE_DIR'],CONFIG['TRAIN_LABEL_DIR'])
    val_ds=AbsoluteDataset(CONFIG['VAL_IMAGE_DIR'],CONFIG['VAL_LABEL_DIR'])
    train_loader=DataLoader(train_ds,batch_size=CONFIG['BATCH_SIZE'],shuffle=True,collate_fn=collate_fn,num_workers=4,pin_memory=True)
    val_loader=DataLoader(val_ds,batch_size=CONFIG['BATCH_SIZE'],shuffle=False,collate_fn=collate_fn,num_workers=4,pin_memory=True)
    # model, opt, sched
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=get_model(CONFIG['NUM_CLASSES']).to(device)
    opt=optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=CONFIG['LEARNING_RATE'])
    sched=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=0.1,patience=10)
    logging.info(f"Config: {json.dumps(CONFIG,indent=2)}")
    # train loop
    best_f1=0;best_epoch=0;no_imp=0;log=[]
    for ep in range(CONFIG['NUM_EPOCHS']):
        model.train();running_loss=0
        for i,(imgs,tgts) in enumerate(train_loader):
            imgs=[i.to(device) for i in imgs]; tgts=[{k:v.to(device) for k,v in t.items()} for t in tgts]
            loss_dict=model(imgs,tgts); loss=sum(loss_dict.values())/CONFIG['accum_steps']; loss.backward();running_loss+=loss.item()
            if (i+1)%CONFIG['accum_steps']==0: opt.step(); opt.zero_grad()
        if (i+1)%CONFIG['accum_steps']!=0: opt.step();opt.zero_grad()
        # eval
        cm=evaluate_model(model,val_loader,device)
        sched.step(cm['overall']['f1'])
        coco=generate_coco_eval(model,val_loader,device,CONFIG['CLASS_NAMES'],CONFIG['run_folder'])
        # log
        entry={'epoch':ep+1,'loss':running_loss,'precision':cm['overall']['precision'],'recall':cm['overall']['recall'],'f1':cm['overall']['f1'],**coco}
        log.append(entry)
        with open(os.path.join(CONFIG['run_folder'],'metrics_log.json'),'w') as f: json.dump(log,f,indent=2)
        logging.info(f"[E{ep+1}] L={running_loss:.4f} P={cm['overall']['precision']:.4f} R={cm['overall']['recall']:.4f} F1={cm['overall']['f1']:.4f}")
        if cm['overall']['f1']>best_f1:
            best_f1=cm['overall']['f1']; best_epoch=ep+1; no_imp=0
            torch.save(model.state_dict(),os.path.join(CONFIG['run_folder'],'best.pth'))
            # summary
            summary={'best_epoch':best_epoch,'best_f1':best_f1,'overall':cm['overall'],'per_class':cm['per_class'],'coco_overall':{k:coco[k] for k in coco if not isinstance(coco[k],dict)},'coco_per_class':coco['per_class_ap'],'config':CONFIG}
            with open(os.path.join(CONFIG['run_folder'],'metrics_summary.json'),'w') as f: json.dump(summary,f,indent=2)
        else: no_imp+=1
        if no_imp>=CONFIG['EARLY_STOPPING_PATIENCE']: logging.info(f"Early stop at E{ep+1}"); break
