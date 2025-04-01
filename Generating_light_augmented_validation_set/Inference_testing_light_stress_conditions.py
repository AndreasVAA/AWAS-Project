from ultralytics import YOLO
from ultralytics import RTDETR
import os
import csv
import time

def run_validation_on_model(model_path, data, imgsz=1280, device="cuda",
                            project="Interference", name="DUM_Testing", **kwargs):
    """
    Run validation on the given data using the specified YOLO model.
    
    Parameters:
      model_path: Path to the YOLO model weights.
      data: YAML file path for data configuration.
      imgsz: Image size (either an int for square images or a tuple).
      device: Device to run inference on (e.g., "cuda").
      project: Base folder for saving outputs.
      name: Name of the run (used as a subfolder within the project folder).
      **kwargs: Additional keyword arguments (e.g., saving parameters).
    
    Returns:
      results: The results from the model's validation/inference.
    """
    #model = YOLO(model_path, task="detect")
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
                             inference_time, num_images, conf_threshold, 
                             csv_filename="metrics_results.csv"):
    """
    Retrieve metrics from the results, print them, and save them to a CSV file.
    Additional information (ground truth count, prediction count, confidence threshold,
    inference speed, and average time per image) are also saved.
    
    The CSV file is saved within the run folder.
    """
    # Retrieve base metrics from the model's mean_results.
    base_metrics = results.mean_results()  
    # Assuming mean_results returns [precision, recall, mAP50, ...]
    precision = base_metrics[0] if len(base_metrics) > 0 else None
    recall    = base_metrics[1] if len(base_metrics) > 1 else None
    mAP50     = base_metrics[2] if len(base_metrics) > 2 else None

    # Use results.box attributes for mAP50-95 and mAP75.
    mAP50_95 = getattr(results.box, "map", None)
    mAP75    = getattr(results.box, "map75", None)

    # Calculate F1 score if precision and recall are available and non-zero.
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = None

    # Calculate additional inference statistics.
    fps = num_images / inference_time if inference_time > 0 else 0
    avg_time_per_image = inference_time / num_images if num_images > 0 else 0

    # Print comparison metrics.
    print(f"Comparison Metrics:")
    print(f"  mAP50-95 (results.box.map): {mAP50_95}")
    print(f"  mAP75 (results.box.map75): {mAP75}")

    # Create a dictionary of metrics and additional info.
    all_metrics = {
        "Precision": precision,
        "Recall": recall,
        "mAP50": mAP50,
        "mAP50-95": mAP50_95,
        "mAP75": mAP75,
        "F1": f1,
        "Ground Truth Count": ground_truth_count,
        "Prediction Count - TP + FP": predicted_count,
        "Confidence Threshold": conf_threshold,
        "Inference Time (s)": inference_time,
        "FPS": fps,
        "Avg Time per Image (s)": avg_time_per_image
    }

    # Print metrics.
    print("Final Metrics:")
    for key, value in all_metrics.items():
        print(f"  {key}: {value}")

    # Ensure CSV is saved within the run folder.
    csv_path = os.path.join(run_folder, csv_filename)
    os.makedirs(run_folder, exist_ok=True)

    # Save metrics to CSV.
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])
        for key, value in all_metrics.items():
            writer.writerow([key, value])
    print(f"Metrics saved to {csv_path}")

def count_instances_from_labels(labels_dir):
    """
    Count non-empty lines in all .txt files in a given labels directory.
    Each non-empty line is counted as one instance.
    
    Returns:
      count: Total number of instances found.
    """
    count = 0
    if os.path.isdir(labels_dir):
        for file in os.listdir(labels_dir):
            if file.endswith(".txt"):
                with open(os.path.join(labels_dir, file), "r") as f:
                    for line in f:
                        if line.strip():
                            count += 1
    return count

def run_and_process_inference(model_path, data, imgsz, device, project, run_name,
                              save_params, gt_labels_dir=None, csv_filename="metrics_results.csv"):
    """
    Run the inference for a given run and process the results.
    This includes timing, calculating FPS, processing metrics, and counting
    predicted instances.
    
    If a ground truth labels directory is provided, its instance count will be computed.
    
    Returns:
      results: The inference results.
    """
    print(f"\nRunning inference on {run_name}...")
    start_time = time.time()
    results = run_validation_on_model(
        model_path=model_path,
        data=data,
        imgsz=imgsz,
        device=device,
        project=project,
        name=run_name,
        **save_params
    )
    inference_time = time.time() - start_time
    print(f"Inference completed in {inference_time:.2f} seconds.")

    # Count predicted instances by reading the labels folder generated by the inference run.
    pred_labels_dir = os.path.join(project, run_name, "labels")
    predicted_count = count_instances_from_labels(pred_labels_dir)
    if predicted_count:
        print(f"Total predicted object instances: {predicted_count}")
    else:
        print(f"No predicted labels folder found at {pred_labels_dir}")

    # Count ground truth instances if a directory is provided.
    if gt_labels_dir:
        ground_truth_count = count_instances_from_labels(gt_labels_dir)
        print(f"Total ground truth label instances: {ground_truth_count}")
    else:
        ground_truth_count = None

    # Determine the number of images processed (by counting .txt files in the predicted labels folder).
    if os.path.isdir(pred_labels_dir):
        num_images = len([f for f in os.listdir(pred_labels_dir) if f.endswith(".txt")])
    else:
        num_images = 0

    # The run folder where metrics CSV will be saved.
    run_folder = os.path.join(project, run_name)
    
    # Get confidence threshold from save_params (if provided).
    conf_threshold = save_params.get('conf', None)

    # Process metrics and save to CSV.
    process_metrics_and_save(results, run_folder, ground_truth_count, predicted_count,
                             inference_time, num_images, conf_threshold,
                             csv_filename=csv_filename)

    return results

# Ground truth counting function (already functional).
def count_ground_truth_instances(labels_dir):
    """
    Count the total ground truth label instances in the given directory.
    
    Returns:
      count: Total ground truth label instances.
    """
    return count_instances_from_labels(labels_dir)

if __name__ == "__main__":

    ## Now set to RT_DETR_MODEL - switch to YOLO in train setup above

    # ----------------------------
    # Configuration
    # ----------------------------
    
    model_path = "/home/itk/Desktop/Andreas/AWAS-Project/RT_DETR_MODEL/runs/RT_DETR_TESTING/weights/best.pt"
    imgsz = 1280  # For non-square images, use a tuple e.g., (1280, 960)
    device = "cuda"
    project = "Interference_RT_DETR_1280"
    base_name = "Default_augmented_model"

    # Common saving parameters (ensure these keys are supported by your YOLO version)
    common_save_params = {
        'save_txt': True,
        'save_conf': True,
        'conf': 0.4,
        'plots': True,
    }

    # ----------------------------
    # Run inference on normal dataset
    # ----------------------------
    normal_data = "/home/itk/Desktop/Andreas/AWAS-Project/Generating_light_augmented_validation_set/dataConf.yaml"
    normal_run_name = f"{base_name}_normal_val"
    normal_gt_labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/Generating_light_augmented_validation_set/val_augmetnted_lightning_conditions/labels"
    normal_results = run_and_process_inference(
        model_path=model_path,
        data=normal_data,
        imgsz=imgsz,
        device=device,
        project=project,
        run_name=normal_run_name,
        save_params=common_save_params,
        gt_labels_dir=normal_gt_labels_dir,
        csv_filename="metrics_results_normal.csv"  # Filename includes 'normal'
    )

    # ----------------------------
    # Run inference on augmented dataset
    # ----------------------------
    augmented_data = "/home/itk/Desktop/Andreas/AWAS-Project/Generating_light_augmented_validation_set/dataConf_light_augmented_dataset.yaml"
    augmented_run_name = f"{base_name}_augmented_val"
    augmented_gt_labels_dir = normal_gt_labels_dir  # Adjust if necessary.
    augmented_results = run_and_process_inference(
        model_path=model_path,
        data=augmented_data,
        imgsz=imgsz,
        device=device,
        project=project,
        run_name=augmented_run_name,
        save_params=common_save_params,
        gt_labels_dir=augmented_gt_labels_dir,
        csv_filename="metrics_results_augmented.csv"  # Filename includes 'augmented'
    )
