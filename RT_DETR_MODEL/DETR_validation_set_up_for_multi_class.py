import os
import csv
import yaml
from ultralytics import RTDETR # Changed from YOLO
import torch
import pandas as pd
import numpy as np

# --- HELPER FUNCTIONS ---

def load_training_args(args_yaml_path):
    """
    Load specified training arguments from the args.yaml file.
    """
    default_args = {
        'model': None, 'data': None, 'epochs': None, 'batch': None,
        'imgsz': None, 'optimizer': None, 'lr0': None, 'lrf': None
    }
    if not os.path.exists(args_yaml_path):
        print(f"Warning: Training args file not found at {args_yaml_path}. Using defaults and recording path as N/A.")
        return default_args
    try:
        with open(args_yaml_path, 'r') as f:
            train_args_loaded = yaml.safe_load(f)

        for key in default_args:
            if key not in train_args_loaded:
                train_args_loaded[key] = None
        return train_args_loaded
    except Exception as e:
        print(f"Error loading training args from {args_yaml_path}: {e}")
        return default_args

def run_rtdetr_validation(model_path, data_yaml, imgsz_val, device, project_dir, run_name, conf_thres, iou_thres, **kwargs):
    """
    Run validation on the given data using the specified RTDETR model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")

    model = RTDETR(model_path) # Use RTDETR
    print(f"Running RT-DETR validation for {model_path} on {data_yaml} with imgsz={imgsz_val}, conf={conf_thres}, iou={iou_thres}")

    results = model.val(
        data=data_yaml,
        imgsz=imgsz_val,
        device=device,
        project=project_dir,
        name=run_name,
        conf=conf_thres,
        iou=iou_thres,
        save_json=True,
        # save_hybrid is not a standard ultralytics val argument, if it causes error, remove or check your specific version.
        # save_hybrid=False, # Generally, this can be removed or handled by **kwargs if needed by a specific version/model
        plots=True,
        **kwargs
    )
    return results

def save_metrics_to_summary_csv_key_value_format(metrics_data, csv_filepath, class_names_list_for_ordering_if_needed):
    """
    Save metrics to a CSV file in a key-value pair format for a single run.
    """
    try:
        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)

        with open(csv_filepath, mode='w', newline='') as csv_file: # 'w' to overwrite for a single run's summary
            writer = csv.writer(csv_file)
            writer.writerow(["Metric", "Value"]) # Header

            ordered_base_keys = [
                "RunName", "Timestamp", "ModelPath", "DataYAML",
                "TrainingImgSz", "ValidationImgSz", "TrainingBatch", "TrainingOptimizer",
                "TrainingLR0", "TrainingLRF", "TrainingEpochs", "TrainingModelDef",
                "Val_Conf_Threshold", "Val_IoU_Threshold",
                "Precision", "Recall", "F1_Score", "mAP50", "mAP75", "mAP50_95"
            ]

            for key in ordered_base_keys:
                if key in metrics_data:
                    writer.writerow([key, metrics_data.get(key, "N/A")])

            if 'per_class_mAP50' in metrics_data and metrics_data['per_class_mAP50']:
                writer.writerow(["--- Per-Class mAP50 ---", "---"]) # Separator
                # Use provided class_names_list_for_ordering_if_needed or sort keys from the dict
                sorted_class_names = class_names_list_for_ordering_if_needed if class_names_list_for_ordering_if_needed else sorted(metrics_data['per_class_mAP50'].keys())

                for class_name in sorted_class_names:
                    if class_name in metrics_data['per_class_mAP50']: # Ensure class name from list is in dict
                        metric_name = f"mAP50_{class_name.replace(' ', '_')}"
                        writer.writerow([metric_name, metrics_data['per_class_mAP50'].get(class_name, "N/A")])

            writer.writerow(["--- Other Metrics ---", "---"]) # Separator
            existing_keys_written = set(ordered_base_keys)
            if 'per_class_mAP50' in metrics_data:
                for class_name in metrics_data['per_class_mAP50'].keys():
                    existing_keys_written.add(f"mAP50_{class_name.replace(' ', '_')}")

            for key, value in metrics_data.items():
                if key not in ordered_base_keys and key != 'per_class_mAP50':
                    if key not in existing_keys_written:
                        writer.writerow([key, value if value is not None else "N/A"])

        print(f"Metrics summary (key-value format) for this run saved to {csv_filepath}")

    except Exception as e:
        print(f"Error writing key-value CSV to {csv_filepath}: {e}")


def process_and_collate_results(results, training_args, model_input_dir_name, model_path_abs,
                                data_yaml_abs, val_imgsz, val_conf, val_iou):
    """
    Extracts comprehensive metrics from RT-DETR (or YOLO) results,
    including overall and per-class P, R, F1, mAP50, mAP75, mAP50-95,
    and collates them with training args.
    """
    if not results:
        print("Error: Results object is None.")
        return None, []

    # --- Initialize Overall Metrics ---
    overall_map50_95, overall_map50, overall_map75 = None, None, None
    overall_precision, overall_recall, overall_f1_score = None, None, None

    class_names_map = getattr(results, 'names', {})
    class_names_list = [class_names_map[i] for i in sorted(class_names_map.keys())] if class_names_map else []

    if hasattr(results, 'box') and results.box is not None:
        # Overall mAP metrics are usually direct attributes of results.box
        overall_map50_95 = getattr(results.box, "map", None)    # mAP50-95
        overall_map50 = getattr(results.box, "map50", None)  # mAP50
        overall_map75 = getattr(results.box, "map75", None)  # mAP75

        # For overall Precision and Recall, Ultralytics results often store them in a 'metrics' dict
        # or they might be the P/R values associated with the 'all' class summary printed to console.
        # The console output `all ... P R mAP50 mAP50-95` gives these values.
        # For instance, 'P' from that line. Let's try to get them from common spots.
        
        # Attempt 1: From results.box.metrics (common for newer ultralytics versions)
        box_metrics_dict = getattr(results.box, 'metrics', None)
        if box_metrics_dict:
            overall_precision = box_metrics_dict.get('metrics/precision(B)', overall_precision)
            overall_recall = box_metrics_dict.get('metrics/recall(B)', overall_recall)
            # If mAPs were not found directly, try from metrics dict too
            if overall_map50_95 is None:
                overall_map50_95 = box_metrics_dict.get('metrics/mAP50-95(B)', overall_map50_95)
            if overall_map50 is None:
                overall_map50 = box_metrics_dict.get('metrics/mAP50(B)', overall_map50)
            if overall_map75 is None:
                overall_map75 = box_metrics_dict.get('metrics/mAP75(B)', overall_map75)
        
        # Attempt 2: If P and R are still None, and if there's only one class or a clear 'all' class P/R
        # The console output values (like P=0.888, R=0.861) are what we want.
        # These often correspond to results.box.p[0] and results.box.r[0] if per-class arrays exist
        # or if there's an overall P and R directly on results.box (less common for direct P/R attributes).
        # The previous console output showed these were available.
        # For `ultralytics==8.3.121`, the values for "all" class P and R in the console are often the primary values.
        # We'll rely on per-class arrays and then derive overall if needed, or use the metrics dict.
        # If there's a single class, its P/R *is* the overall P/R.
        if len(class_names_list) == 1:
            per_class_p_array = getattr(results.box, 'p', None)
            per_class_r_array = getattr(results.box, 'r', None)
            if overall_precision is None and per_class_p_array is not None and len(per_class_p_array) == 1:
                overall_precision = per_class_p_array[0]
            if overall_recall is None and per_class_r_array is not None and len(per_class_r_array) == 1:
                overall_recall = per_class_r_array[0]

        if overall_precision is not None and overall_recall is not None:
            if (overall_precision + overall_recall) > 0:
                overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
            else:
                overall_f1_score = 0.0
    else:
        print("Warning: 'results.box' not found or is None. Cannot extract most detection metrics.")
        # Fallback to results.metrics if it exists (less common for detection, more for classification)
        if hasattr(results, 'metrics') and results.metrics is not None:
            print("Info: Checking 'results.metrics' as a fallback.")
            metrics_dict = results.metrics
            overall_precision = metrics_dict.get('precision', overall_precision)
            overall_recall = metrics_dict.get('recall', overall_recall)
            if overall_precision is not None and overall_recall is not None and overall_f1_score is None:
                overall_f1_score = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
            if overall_map50_95 is None: overall_map50_95 = metrics_dict.get('mAP50-95(B)', metrics_dict.get('map', None))
            if overall_map50 is None: overall_map50 = metrics_dict.get('mAP50(B)', metrics_dict.get('map50', None))
            if overall_map75 is None: overall_map75 = metrics_dict.get('mAP75(B)', metrics_dict.get('map75', None))
        else:
            print("Error: No 'results.box' or 'results.metrics' available. Cannot extract detailed metrics.")
            return None, []


    collated_data = {
        "RunName": f"{model_input_dir_name}_val_c{val_conf}_i{val_iou}",
        "Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "ModelPath": model_path_abs, "DataYAML": data_yaml_abs,
        "TrainingImgSz": training_args.get('imgsz', 'N/A'), "ValidationImgSz": val_imgsz,
        "TrainingBatch": training_args.get('batch', 'N/A'), "TrainingOptimizer": training_args.get('optimizer', 'N/A'),
        "TrainingLR0": training_args.get('lr0', 'N/A'), "TrainingLRF": training_args.get('lrf', 'N/A'),
        "TrainingEpochs": training_args.get('epochs', 'N/A'), "TrainingModelDef": training_args.get('model', 'N/A'),
        "Val_Conf_Threshold": val_conf, "Val_IoU_Threshold": val_iou,
        "Precision_Overall": round(overall_precision, 5) if overall_precision is not None else 'N/A',
        "Recall_Overall": round(overall_recall, 5) if overall_recall is not None else 'N/A',
        "F1_Score_Overall": round(overall_f1_score, 5) if overall_f1_score is not None else 'N/A',
        "mAP50_Overall": round(overall_map50, 5) if overall_map50 is not None else 'N/A',
        "mAP75_Overall": round(overall_map75, 5) if overall_map75 is not None else 'N/A',
        "mAP50-95_Overall": round(overall_map50_95, 5) if overall_map50_95 is not None else 'N/A',
    }

    # --- Per-Class Metrics Extraction ---
    if hasattr(results, 'box') and results.box is not None and class_names_list:
        # These are typically numpy arrays with one value per class
        per_class_p_array = getattr(results.box, 'p', None)      # Per-class Precision array
        per_class_r_array = getattr(results.box, 'r', None)      # Per-class Recall array
        per_class_f1_array = getattr(results.box, 'f1', None)     # Per-class F1-score array
        
        # results.box.all_ap is typically a 2D array: (num_classes, num_iou_thresholds)
        # IoU thresholds are usually 0.50, 0.55, ..., 0.95 (10 thresholds)
        # Index 0: IoU 0.50 (for mAP50)
        # Index 5: IoU 0.75 (for mAP75, if IoU step is 0.05)
        all_ap_values = getattr(results.box, 'all_ap', None) 

        # results.box.maps is often per-class mAP50-95 for YOLO/RT-DETR
        per_class_map50_95_array = getattr(results.box, 'maps', None) 

        for i, class_name in enumerate(class_names_list):
            safe_class_name = class_name.replace(' ', '_') # For consistent key naming

            # Per-class Precision, Recall, F1
            if per_class_p_array is not None and isinstance(per_class_p_array, np.ndarray) and i < len(per_class_p_array):
                collated_data[f"P_{safe_class_name}"] = round(float(per_class_p_array[i]), 5)
            if per_class_r_array is not None and isinstance(per_class_r_array, np.ndarray) and i < len(per_class_r_array):
                collated_data[f"R_{safe_class_name}"] = round(float(per_class_r_array[i]), 5)
            if per_class_f1_array is not None and isinstance(per_class_f1_array, np.ndarray) and i < len(per_class_f1_array):
                collated_data[f"F1_{safe_class_name}"] = round(float(per_class_f1_array[i]), 5)
            
            # Per-class mAP50, mAP75 from all_ap array
            if all_ap_values is not None and isinstance(all_ap_values, np.ndarray) and \
               all_ap_values.ndim == 2 and i < all_ap_values.shape[0]:
                
                if all_ap_values.shape[1] > 0: # Check for IoU thresholds (mAP50 at index 0)
                    collated_data[f"mAP50_{safe_class_name}"] = round(float(all_ap_values[i, 0]), 5)
                
                # Index for mAP75: If IoUs are 0.5:0.05:0.95, then 0.75 is at index (0.75-0.5)/0.05 = 5
                if all_ap_values.shape[1] > 5: 
                    collated_data[f"mAP75_{safe_class_name}"] = round(float(all_ap_values[i, 5]), 5)

            # Per-class mAP50-95 (often from results.box.maps directly)
            if per_class_map50_95_array is not None and isinstance(per_class_map50_95_array, np.ndarray) and i < len(per_class_map50_95_array):
                collated_data[f"mAP50-95_{safe_class_name}"] = round(float(per_class_map50_95_array[i]), 5)
    
    # --- Print Metrics ---
    print("\n--- Overall Metrics ---")
    print(f"  Precision (Overall): {collated_data.get('Precision_Overall', 'N/A')}")
    print(f"  Recall (Overall): {collated_data.get('Recall_Overall', 'N/A')}")
    print(f"  F1 Score (Overall): {collated_data.get('F1_Score_Overall', 'N/A')}")
    print(f"  mAP50 (Overall): {collated_data.get('mAP50_Overall', 'N/A')}")
    print(f"  mAP75 (Overall): {collated_data.get('mAP75_Overall', 'N/A')}")
    print(f"  mAP50-95 (Overall): {collated_data.get('mAP50-95_Overall', 'N/A')}")

    print("\n--- Per-Class Metrics (if available) ---")
    if not class_names_list:
        print("  No class names found to detail per-class metrics.")
    else:
        for class_name in class_names_list:
            safe_class_name = class_name.replace(' ', '_')
            p_val = collated_data.get(f"P_{safe_class_name}", None)
            r_val = collated_data.get(f"R_{safe_class_name}", None)
            f1_val = collated_data.get(f"F1_{safe_class_name}", None)
            map50_val = collated_data.get(f"mAP50_{safe_class_name}", None)
            map75_val = collated_data.get(f"mAP75_{safe_class_name}", None)
            map50_95_val = collated_data.get(f"mAP50-95_{safe_class_name}", None)

            # Only print if at least one metric for this class was found
            if any(val is not None for val in [p_val, r_val, f1_val, map50_val, map75_val, map50_95_val]):
                print(f"  Class: {class_name}")
                if p_val is not None: print(f"    Precision: {p_val}")
                if r_val is not None: print(f"    Recall: {r_val}")
                if f1_val is not None: print(f"    F1-Score: {f1_val}")
                if map50_val is not None: print(f"    mAP50: {map50_val}")
                if map75_val is not None: print(f"    mAP75: {map75_val}")
                if map50_95_val is not None: print(f"    mAP50-95: {map50_95_val}")
            elif i == 0: # only print once if no per class metrics were found at all
                 print("  No detailed per-class P,R,F1 or mAP metrics were extracted for CSV (check results object structure if expected).")


    return collated_data, class_names_list

# --- MAIN EXECUTION LOGIC ---

def perform_experiment_validations(
    input_dirs_list,
    data_yaml_path,
    output_base_dir,
    val_img_size,
    val_conf_thresh,
    val_iou_thresh,
    device_setting,
    summary_csv_filename
    ):
    os.makedirs(output_base_dir, exist_ok=True)

    for single_input_dir in input_dirs_list:
        print(f"\nProcessing Model Directory: {single_input_dir}")
        print("=====================================================================")

        model_input_dir_name = os.path.basename(os.path.normpath(single_input_dir))
        # Standard weight file name for Ultralytics models
        model_path = os.path.join(single_input_dir, "weights", "best.pt")
        args_yaml_path = os.path.join(single_input_dir, "args.yaml") # Training arguments

        try:
            if not os.path.exists(model_path):
                print(f"ERROR: Model weights 'best.pt' not found in {os.path.join(single_input_dir, 'weights')}. Skipping directory '{single_input_dir}'.")
                continue

            training_params = load_training_args(args_yaml_path)

            ultralytics_project_for_val = output_base_dir
            ultralytics_run_name_for_val = f"{model_input_dir_name}_val_c{val_conf_thresh}_i{val_iou_thresh}"
            specific_run_ultralytics_output_dir = os.path.join(ultralytics_project_for_val, ultralytics_run_name_for_val)

            print(f"Starting validation for model in: {single_input_dir}")
            print(f"  Model path: {model_path}")
            print(f"  Data YAML: {data_yaml_path}")
            print(f"  Ultralytics outputs (plots, etc.) will be in: {specific_run_ultralytics_output_dir}")

            results = run_rtdetr_validation( # Changed to RTDETR validation function
                model_path=model_path,
                data_yaml=data_yaml_path,
                imgsz_val=val_img_size,
                device=device_setting,
                project_dir=ultralytics_project_for_val,
                run_name=ultralytics_run_name_for_val,
                conf_thres=val_conf_thresh,
                iou_thres=val_iou_thresh,
            )

            if results:
                model_path_abs = os.path.abspath(model_path)
                data_yaml_abs = os.path.abspath(data_yaml_path)

                collated_metrics, class_names_list = process_and_collate_results(
                    results, training_params, model_input_dir_name,
                    model_path_abs, data_yaml_abs, val_img_size,
                    val_conf_thresh, val_iou_thresh
                )
                if collated_metrics:
                    os.makedirs(specific_run_ultralytics_output_dir, exist_ok=True)
                    summary_csv_path_for_this_run = os.path.join(specific_run_ultralytics_output_dir, summary_csv_filename)
                    # Corrected function name for saving CSV
                    save_metrics_to_summary_csv_key_value_format(collated_metrics, summary_csv_path_for_this_run, class_names_list)
                    print(f"Validation for '{model_input_dir_name}' complete.")
                else:
                    print(f"Could not process or collate results for '{model_input_dir_name}'.")
            else:
                print(f"Validation run did not produce expected results for '{model_input_dir_name}'.")

        except FileNotFoundError as fnf_error:
            print(f"ERROR processing '{single_input_dir}': Required file not found. {fnf_error}")
        except Exception as e:
            print(f"An unexpected error occurred during validation of '{model_input_dir_name}': {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("---------------------------------------------------------------------")

    print(f"\nAll specified model directories processed. Individual metric summaries are saved within their respective run folders inside '{output_base_dir}'.")

# --- SCRIPT ENTRY POINT & CONFIGURATION ---
if __name__ == "__main__":

    # --- CONFIGURABLE PARAMETERS ---
    INPUT_DIRS_LIST = [
        # Example: "C:/Path/To/Your/RTDETR_Runs/exp1_rtdetr_coco",
        # Example: "/home/user/rtdetr_experiments/exp_rtdetr_custom_data",
        # Update with your actual RT-DETR model directories
        "/home/itk/Desktop/Andreas/AWAS-Project/RT_DETR_MODEL/Runing_new_config_more_similar_to_paper/RT_DETR_1280_batch2_lr0=0.0002",
    ]
    # Ensure this data.yaml is compatible with your RT-DETR model and dataset
    DATA_YAML_PATH = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml" # For single class
    #DATA_YAML_PATH = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf_multiple_Classes.yaml" # For multi-class
    
    OUTPUT_BASE_DIR_PATH = "./NEW_rtdetr_validation_results_WITH_VERIFIED_CONF_TO_DEFAULT" # Changed output directory name
    SUMMARY_CSV_FILENAME = "rtdetr_val_metrics_summary.csv" # Changed CSV filename
    
    VAL_IMAGE_SIZE = 1280
    VAL_CONF_THRESHOLD = 0.01 # RT-DETR often works well with low confidence thresholds during validation
    VAL_IOU_THRESHOLD = 0.7   # Common default, adjust as needed
    DEVICE_SETTING = "cuda" if torch.cuda.is_available() else "cpu"
    # --- END OF CONFIGURABLE PARAMETERS ---

    try:
        _ = pd.Timestamp.now()
        _ = torch.cuda.is_available()
    except NameError:
        print("Import error: Critical libraries (pandas, torch, pyyaml, ultralytics) might be missing or not imported correctly.")
        print("Please ensure they are installed in your Python environment.")
        exit(1)
    except Exception as e:
        print(f"An issue occurred with library checks: {e}")

    if not INPUT_DIRS_LIST or not DATA_YAML_PATH or \
       (len(INPUT_DIRS_LIST) > 0 and (INPUT_DIRS_LIST[0].startswith("C:/Path/To/Your") or INPUT_DIRS_LIST[0].startswith("/home/user/rtdetr_experiments"))) or \
       DATA_YAML_PATH.startswith("path/to/your"):
        print("\nWARNING: Please update the placeholder paths in the 'CONFIGURABLE PARAMETERS' section "
              "of the script (INPUT_DIRS_LIST, DATA_YAML_PATH) before running.")

    if INPUT_DIRS_LIST and not (len(INPUT_DIRS_LIST) == 1 and (INPUT_DIRS_LIST[0].startswith("C:/Path/To/Your") or INPUT_DIRS_LIST[0].startswith("/home/user/rtdetr_experiments"))):
        perform_experiment_validations(
            input_dirs_list=INPUT_DIRS_LIST,
            data_yaml_path=DATA_YAML_PATH,
            output_base_dir=OUTPUT_BASE_DIR_PATH,
            val_img_size=VAL_IMAGE_SIZE,
            val_conf_thresh=VAL_CONF_THRESHOLD,
            val_iou_thresh=VAL_IOU_THRESHOLD,
            device_setting=DEVICE_SETTING,
            summary_csv_filename=SUMMARY_CSV_FILENAME
        )
    elif not INPUT_DIRS_LIST:
        print("\nINFO: INPUT_DIRS_LIST is empty. No models to validate. Please configure it in the script.")
    else:
        print("\nINFO: Script not run due to placeholder paths. Please update INPUT_DIRS_LIST.")