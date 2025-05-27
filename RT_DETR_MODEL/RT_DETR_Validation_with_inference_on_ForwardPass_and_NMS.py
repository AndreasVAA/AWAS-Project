from ultralytics import RTDETR
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
    Ultralytics RT‑DETR high‑level validation to compute detection metrics.
    """
    model = RTDETR(model_path)
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
                             num_images_validated, conf_threshold, csv_filename="metrics_results.csv",
                             validation_imgsz=None, train_imgsz=None, lr0=None, lrf=None, optimizer=None, tr_batch=None):
    """
    Save both detection and core timing (forward+NMS) metrics to CSV,
    including parameters from training args.yaml.
    """
    # Detection metrics
    base      = results.mean_results()
    precision = base[0] if len(base)>0 else None
    recall    = base[1] if len(base)>1 else None
    mAP50     = base[2] if len(base)>2 else None
    mAP50_95  = getattr(results.box, "map", None)
    mAP75     = getattr(results.box, "map75", None)
    f1        = 2*precision*recall/(precision+recall) if precision and recall and (precision+recall) > 0 else None

    # Core timing + counts + training params
    metrics = {
        "Precision": precision,
        "Recall": recall,
        "mAP50": mAP50,
        "mAP50-95": mAP50_95,
        "mAP75": mAP75,
        "F1": f1,
        "Ground Truth Count": ground_truth_count,
        "Prediction Count - TP + FP": predicted_count,
        "Confidence Threshold Used": conf_threshold, # Clarified name
        "Validation Image Size": validation_imgsz,
        "Num Images Validated": num_images_validated,
        "Core GPU Time (s)": core_gpu_time,
        "Core Avg Time per Image (ms)": core_avg_ms,
        "Core FPS": core_fps,
        "Training Image Size (from args.yaml)": train_imgsz,
        "Training lr0 (from args.yaml)": lr0,
        "Training lrf (from args.yaml)": lrf,
        "Training Optimizer (from args.yaml)": optimizer,
        "Training Batch (from args.yaml)": tr_batch, # Corrected "BAtch" to "Batch" and using the passed tr_batch
    }

    os.makedirs(run_folder, exist_ok=True)
    csv_path = os.path.join(run_folder, csv_filename)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric","Value"])
        for k, v in metrics.items():
            writer.writerow([k, v if v is not None else "N/A"])
    print(f"Metrics saved to {csv_path}")

def count_instances_from_labels(labels_dir):
    """Count all lines in .txt files under labels_dir as ground‑truth instances."""
    count = 0
    if os.path.isdir(labels_dir):
        for fn in os.listdir(labels_dir):
            if fn.endswith(".txt"):
                with open(os.path.join(labels_dir, fn)) as f:
                    for line in f:
                        if line.strip():
                            count += 1
    return count

def count_predictions_with_conf(labels_dir, conf_threshold):
    """
    Count predicted instances from labels_dir whose confidence ≥ conf_threshold.
    Each .txt line: <class> <x> <y> <w> <h> <confidence>
    """
    count = 0
    if os.path.isdir(labels_dir):
        for fname in os.listdir(labels_dir):
            if not fname.endswith(".txt"):
                continue
            with open(os.path.join(labels_dir, fname)) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        try:
                            conf = float(parts[-1])
                        except ValueError:
                            continue
                        if conf >= conf_threshold:
                            count += 1
    return count

def run_and_process_inference(model_path, data_yaml_path, validation_imgsz, device_str,
                              project_name, run_name_str,
                              save_params_dict, gt_labels_dir_path=None, csv_filename_str="validation_metrics.csv",
                              train_imgsz_from_yaml=None, lr0_from_yaml=None, lrf_from_yaml=None, optimizer_from_yaml=None, tr_batch_from_yaml=None): # Added tr_batch_from_yaml
    print(f"\nRunning inference on {run_name_str} (forward + NMS timing only)…")
    selected_device = torch.device(device_str)
    rtdetr_model    = RTDETR(model_path)
    core_model      = rtdetr_model.model.to(selected_device).eval()

    conf_thres_for_timing_and_counting = save_params_dict.get("conf", 0.25)
    iou_thres_for_timing = save_params_dict.get("iou", 0.5)

    with open(data_yaml_path, "r") as f:
        cfg_data   = yaml.safe_load(f)
    val_img_dir    = cfg_data["val"]
    img_paths_list  = sorted(glob.glob(os.path.join(val_img_dir, "*.jpg")) +
                             glob.glob(os.path.join(val_img_dir, "*.png")) +
                             glob.glob(os.path.join(val_img_dir, "*.jpeg")))
    num_images_found = len(img_paths_list)
    if num_images_found == 0:
        raise RuntimeError(f"No images found in {val_img_dir}")

    dummy_input = torch.zeros((1,3,validation_imgsz,validation_imgsz), device=selected_device)
    starter_event = torch.cuda.Event(enable_timing=True)
    ender_event   = torch.cuda.Event(enable_timing=True)
    with torch.no_grad():
        for _ in range(10):
            preds_warmup = core_model(dummy_input)
            _ = non_max_suppression(preds_warmup[0] if isinstance(preds_warmup, tuple) else preds_warmup,
                                    conf_thres_for_timing_and_counting, iou_thres_for_timing)
            torch.cuda.synchronize()

    times_list = []
    with torch.no_grad():
        for img_path_single in img_paths_list:
            img_orig = cv2.imread(img_path_single)
            img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (validation_imgsz, validation_imgsz))

            tensor_input = (torch.from_numpy(img_resized)
                                .float()
                                .permute(2,0,1)
                                .div(255.0)
                                .unsqueeze(0)
                                .to(selected_device))

            torch.cuda.synchronize()
            starter_event.record()

            preds_infer = core_model(tensor_input)
            _ = non_max_suppression(preds_infer[0] if isinstance(preds_infer, tuple) else preds_infer,
                                    conf_thres_for_timing_and_counting, iou_thres_for_timing)

            ender_event.record()
            torch.cuda.synchronize()
            times_list.append(starter_event.elapsed_time(ender_event) / 1000.0)

    core_gpu_total_time = sum(times_list)
    core_avg_ms_per_image = (core_gpu_total_time / num_images_found) * 1000.0 if num_images_found > 0 else 0
    core_fps_calc = num_images_found / core_gpu_total_time if core_gpu_total_time > 0 else 0

    val_results = run_validation_on_model(
        model_path=model_path,
        data=data_yaml_path,
        imgsz=validation_imgsz,
        device=selected_device.type,
        project=project_name,
        name=run_name_str,
        **save_params_dict
    )

    current_run_folder = os.path.join(project_name, run_name_str)
    pred_labels_dir_path = os.path.join(current_run_folder, "labels")
    predicted_count_val = count_predictions_with_conf(pred_labels_dir_path, conf_thres_for_timing_and_counting)
    gt_count_val = count_instances_from_labels(gt_labels_dir_path) if gt_labels_dir_path else None

    process_metrics_and_save(
        val_results, current_run_folder, gt_count_val, predicted_count_val,
        core_gpu_total_time, core_avg_ms_per_image, core_fps_calc,
        num_images_found, conf_thres_for_timing_and_counting,
        csv_filename_str,
        validation_imgsz=validation_imgsz,
        train_imgsz=train_imgsz_from_yaml,
        lr0=lr0_from_yaml,
        lrf=lrf_from_yaml,
        optimizer=optimizer_from_yaml,
        tr_batch=tr_batch_from_yaml
    )

    print("Core forward+NMS timing complete. Detection metrics appended to CSV.")
    return val_results

if __name__ == "__main__":
    input_training_folder = "/home/itk/Desktop/Andreas/AWAS-Project/RT_DETR_MODEL/Runing_new_config_more_similar_to_paper/RT_DETR_1280_batch2_lr0=0.0002"

    model_file_path = os.path.join(input_training_folder, "weights", "best.pt")
    args_yaml_file_path = os.path.join(input_training_folder, "args.yaml")

    if not os.path.isfile(model_file_path):
        raise FileNotFoundError(f"Model file not found: {model_file_path}")
    if not os.path.isfile(args_yaml_file_path):
        print(f"Warning: args.yaml not found at {args_yaml_file_path}. Some training parameters will be missing in the CSV.")
        training_args_dict = {}
    else:
        with open(args_yaml_file_path, 'r') as f:
            training_args_dict = yaml.safe_load(f)

    train_imgsz_from_args = training_args_dict.get('imgsz')
    train_lr0_from_args = training_args_dict.get('lr0')
    train_lrf_from_args = training_args_dict.get('lrf')
    train_optimizer_from_args = training_args_dict.get('optimizer')
    train_batch_from_args = training_args_dict.get('batch') # <-- Added: Extract batch size (usually 'batch' key)

    validation_run_imgsz = 1280
    device_to_use = "cuda:0"
    validation_project_folder = "Lower_learning_rater_validation_results"
    input_folder_basename = os.path.basename(input_training_folder.rstrip('/\\'))
    current_run_name = f"{input_folder_basename}conf=0.01ValAt{validation_run_imgsz}"

    val_common_params = {
        'save_txt': True,
        'save_conf': True,
        'conf': 0.01,
        'iou': 0.6,
        'plots': True,
    }

    data_config_yaml = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml"
    ground_truth_labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels"

    print(f"Starting validation for model from: {input_training_folder}")
    print(f"Model: {model_file_path}")
    print(f"Using validation image size: {validation_run_imgsz}")
    print(f"Output project/run: {validation_project_folder}/{current_run_name}")
    print(f"Params for model.val(): conf={val_common_params['conf']}, iou={val_common_params['iou']}")
    # Updated print statement to include batch size
    print(f"Training args loaded: imgsz={train_imgsz_from_args}, batch={train_batch_from_args}, lr0={train_lr0_from_args}, lrf={train_lrf_from_args}, optimizer={train_optimizer_from_args}")

    run_and_process_inference(
        model_path=model_file_path,
        data_yaml_path=data_config_yaml,
        validation_imgsz=validation_run_imgsz,
        device_str=device_to_use,
        project_name=validation_project_folder,
        run_name_str=current_run_name,
        save_params_dict=val_common_params,
        gt_labels_dir_path=ground_truth_labels_dir,
        csv_filename_str="validation_and_timing_metrics.csv",
        train_imgsz_from_yaml=train_imgsz_from_args,
        lr0_from_yaml=train_lr0_from_args,
        lrf_from_yaml=train_lrf_from_args,
        optimizer_from_yaml=train_optimizer_from_args,
        tr_batch_from_yaml=train_batch_from_args # <-- Added: Pass the extracted batch size
    )
    print("Validation completed. Results saved in the project folder.")