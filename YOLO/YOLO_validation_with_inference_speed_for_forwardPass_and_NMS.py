from ultralytics import YOLO
import os
import csv
import time
import yaml
import glob
import cv2
import torch
from ultralytics.utils.ops import non_max_suppression

def run_validation_on_model(model_path, data, imgsz=1280, device="cuda",
                            project="Interference", name="DUM_Testing", **kwargs):
    """
    Ultralytics high‑level validation to compute detection metrics.
    """
    model = YOLO(model_path, task="detect")
    results = model.val(
        data=data,
        imgsz=imgsz,
        device=device,
        project=project,
        name=name,
        **kwargs
    )
    return results

def process_metrics_and_save(results, run_folder, ground_truth_count, predicted_count, 
                             core_gpu_time, core_avg_ms, core_fps,
                             num_images, conf_threshold, csv_filename="metrics_results.csv"):
    """
    Save both detection and core timing (forward+NMS) metrics to CSV.
    """
    # 1) Detection metrics
    base      = results.mean_results()  
    precision = base[0] if len(base)>0 else None
    recall    = base[1] if len(base)>1 else None
    mAP50     = base[2] if len(base)>2 else None
    mAP50_95  = getattr(results.box, "map", None)
    mAP75     = getattr(results.box, "map75", None)
    f1        = 2*precision*recall/(precision+recall) if precision and recall else None

    # 2) Core timing metrics
    metrics = {
        # detection
        "Precision": precision,
        "Recall": recall,
        "mAP50": mAP50,
        "mAP50-95": mAP50_95,
        "mAP75": mAP75,
        "F1": f1,
        # counts
        "Ground Truth Count": ground_truth_count,
        "Prediction Count - TP + FP": predicted_count,
        "Confidence Threshold": conf_threshold,
        # core timing
        "Core GPU Time (s)": core_gpu_time,
        "Core Avg Time per Image (ms)": core_avg_ms,
        "Core FPS": core_fps,
        # images processed
        "Num Images": num_images
    }

    os.makedirs(run_folder, exist_ok=True)
    csv_path = os.path.join(run_folder, csv_filename)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric","Value"])
        for k, v in metrics.items():
            writer.writerow([k, v])
    print(f"Metrics saved to {csv_path}")

def count_instances_from_labels(labels_dir):
    count = 0
    if os.path.isdir(labels_dir):
        for fn in os.listdir(labels_dir):
            if fn.endswith(".txt"):
                with open(os.path.join(labels_dir, fn)) as f:
                    for line in f:
                        if line.strip():
                            count += 1
    return count

def run_and_process_inference(model_path, data, imgsz, device, project, run_name,
                              save_params, gt_labels_dir=None, csv_filename="validation_metrics.csv"):
    print(f"\nRunning inference on {run_name} (forward + NMS timing only)…")
    device     = torch.device(device)
    yolo       = YOLO(model_path, task="detect")
    core       = yolo.model.to(device).eval()
    conf_thres = save_params.get("conf", 0.25)
    iou_thres  = save_params.get("iou", 0.5)

    # 1) Gather all validation images
    with open(data, "r") as f:
        cfg     = yaml.safe_load(f)
    val_dir    = cfg["val"]
    img_paths  = sorted(glob.glob(os.path.join(val_dir, "*.jpg")) +
                        glob.glob(os.path.join(val_dir, "*.png")))
    num_images = len(img_paths)
    if num_images == 0:
        raise RuntimeError(f"No images found in {val_dir}")

    # 2) Warm‑up: forward + NMS
    dummy   = torch.zeros((1,3,imgsz,imgsz), device=device)
    starter = torch.cuda.Event(enable_timing=True)
    ender   = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for _ in range(10):
            preds = core(dummy)
            _     = non_max_suppression(preds, conf_thres, iou_thres)
            torch.cuda.synchronize()

    # 3) Time forward + NMS per image
    times = []
    with torch.no_grad():
        for path in img_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Load one image and unsqueeze to batch size = 1
            tensor = (torch.from_numpy(img)
                      .float()
                      .permute(2,0,1)
                      .div(255.0)
                      .unsqueeze(0) # batch=1
                      .to(device))

            torch.cuda.synchronize()
            starter.record()

            preds = core(tensor)  
            _     = non_max_suppression(preds, conf_thres, iou_thres)

            ender.record()
            torch.cuda.synchronize()

            times.append(starter.elapsed_time(ender) / 1000.0)

    core_gpu_time = sum(times)
    core_avg_ms   = (core_gpu_time / num_images) * 1000.0
    core_fps      = num_images / core_gpu_time if core_gpu_time > 0 else 0

    # 4) High‑level .val() for detection metrics
    results = run_validation_on_model(
        model_path=model_path,
        data=data,
        imgsz=imgsz,
        device=device.type,
        project=project,
        name=run_name,
        **save_params
    )

    # 5) Count predicted / ground‑truth instances
    run_folder      = os.path.join(project, run_name)
    pred_labels_dir = os.path.join(run_folder, "labels")
    predicted_count = count_instances_from_labels(pred_labels_dir)
    gt_count        = count_instances_from_labels(gt_labels_dir) if gt_labels_dir else None

    # 6) Save everything
    process_metrics_and_save(
        results, run_folder, gt_count, predicted_count,
        core_gpu_time, core_avg_ms, core_fps,
        num_images, conf_thres, csv_filename
    )

    print("Core forward+NMS timing complete. Detection metrics appended to CSV.")
    return results

if __name__ == "__main__":
    model_path    = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/runs/YOLO11_modelType_M_1280x960/weights/best.pt"
    imgsz         = 1280
    device        = "cuda"
    project       = "Validation_testing_with_inference_on_forwardPass_and_NMS"
    run_name      = "YOLO11M_1280_SGD_optimizer"
    common_params = {
        'save_txt': True,
        'save_conf': True,
        'conf': 0.4,
        'plots': True,
        'iou': 0.6,          # for NMS
    }
    data_yaml     = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml"
    gt_labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels"

    run_and_process_inference(
        model_path, data_yaml, imgsz, device, project, run_name,
        common_params, gt_labels_dir, csv_filename="validation_metrics.csv"
    )
    print("Validation completed. Results saved in the project folder.")

"""
Core GPU Time (s):
The sum of just the model’s forward pass plus NMS (via ultralytics.utils.ops.non_max_suppression) across all images.

Core Avg Time per Image (ms):
The average of those per-image forward+NMS times, in milliseconds.

Core FPS:
How many images/second the forward + NMS stage can run, at batch size 1.
"""
