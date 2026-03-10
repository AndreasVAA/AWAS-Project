import os
import csv
import yaml
from ultralytics import YOLO # Use YOLO for your YOLO models
import torch 
import pandas as pd
import numpy as np # For isinstance checks

# --- HELPER FUNCTIONS ---

def load_training_args(args_yaml_path):
    default_args = {
        'model': None, 'data': None, 'epochs': None, 'batch': None,
        'imgsz': None, 'optimizer': None, 'lr0': None, 'lrf': None
    }
    if not os.path.exists(args_yaml_path):
        print(f"Warning: Training args file not found at {args_yaml_path}. Using defaults.")
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

def run_yolo_validation(model_path, data_yaml, imgsz_val, device, project_dir, run_name, conf_thres, iou_thres, **kwargs):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")
    model = YOLO(model_path) 
    print(f"Running validation for {model_path} on {data_yaml} with imgsz={imgsz_val}, conf={conf_thres}, iou={iou_thres}")
    results = model.val(
        data=data_yaml, imgsz=imgsz_val, device=device, project=project_dir, name=run_name,      
        conf=conf_thres, iou=iou_thres, save_json=True, save_hybrid=False, plots=True, **kwargs
    )
    return results

# Using the key-value pair CSV format as per your last preference:
def save_metrics_to_summary_csv(metrics_data, csv_filepath, class_names_list):
    """
    Save metrics to a CSV file in a key-value pair format for a single run,
    including detailed per-class metrics.
    """
    try:
        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
        
        with open(csv_filepath, mode='w', newline='') as csv_file: # 'w' to overwrite
            writer = csv.writer(csv_file)
            writer.writerow(["Metric", "Value"]) # Header

            # 1. Experiment Info and Overall Training Params
            exp_info_keys = [
                "RunName", "Timestamp", "ModelPath", "DataYAML", 
                "TrainingImgSz", "ValidationImgSz", "TrainingBatch", "TrainingOptimizer", 
                "TrainingLR0", "TrainingLRF", "TrainingEpochs", "TrainingModelDef",
                "Val_Conf_Threshold", "Val_IoU_Threshold"
            ]
            for key in exp_info_keys:
                if key in metrics_data:
                    writer.writerow([key, metrics_data.get(key, "N/A")])
            
            # 2. Overall Validation Metrics
            writer.writerow(["--- Overall Validation Metrics ---", "---"])
            overall_metrics_keys = [
                "Precision_Overall", "Recall_Overall", "F1_Score_Overall", 
                "mAP50_Overall", "mAP75_Overall", "mAP50-95_Overall"
            ]
            for key in overall_metrics_keys:
                if key in metrics_data:
                    writer.writerow([key, metrics_data.get(key, "N/A")])

            # 3. Per-Class Metrics
            # Group by class name for readability if class_names_list is available
            if class_names_list:
                writer.writerow(["--- Per-Class Metrics ---", "---"])
                for class_name in class_names_list:
                    safe_class_name = class_name.replace(' ', '_')
                    # Write all metrics for this class together
                    per_class_metric_types = {"P": "Precision", "R": "Recall", "F1": "F1-Score", 
                                              "mAP50": "mAP@0.50", "mAP75": "mAP@0.75", "mAP50-95": "mAP@0.5:0.95"}
                    
                    class_header_written = False
                    for prefix, display_name_suffix in per_class_metric_types.items():
                        metric_key = f"{prefix}_{safe_class_name}"
                        if metric_key in metrics_data:
                            if not class_header_written:
                                # writer.writerow([f"--- Metrics for Class: {class_name} ---", "---"]) # Optional sub-header per class
                                class_header_written = True
                            writer.writerow([f"{display_name_suffix} ({class_name})", metrics_data[metric_key]])
            
            # 4. Write any other remaining metrics from metrics_data (fallback)
            # This ensures nothing is missed if new keys are added to collated_data
            # and not covered above.
            written_keys = set(exp_info_keys + overall_metrics_keys)
            if class_names_list:
                for class_name in class_names_list:
                    safe_class_name = class_name.replace(' ', '_')
                    for prefix in ["P", "R", "F1", "mAP50", "mAP75", "mAP50-95"]:
                        written_keys.add(f"{prefix}_{safe_class_name}")
            
            other_metrics_exist_and_to_write = []
            for key, value in metrics_data.items():
                if key not in written_keys:
                    other_metrics_exist_and_to_write.append((key, value))
            
            if other_metrics_exist_and_to_write:
                writer.writerow(["--- Other Extracted Metrics ---", "---"])
                for key, value in sorted(other_metrics_exist_and_to_write): # Sort for consistency
                    writer.writerow([key, value if value is not None else "N/A"])

        print(f"Detailed metrics summary (key-value format) saved to {csv_filepath}")

    except Exception as e:
        print(f"Error writing detailed key-value CSV to {csv_filepath}: {e}")
        import traceback
        traceback.print_exc()

# Corrected process_and_collate_results function
# Make sure numpy is imported at the top of your script:
# import numpy as np

def process_and_collate_results(results, training_args, model_input_dir_name, model_path_abs, 
                                data_yaml_abs, val_imgsz, val_conf, val_iou):
    """
    Extracts metrics from YOLO results, including more per-class details, 
    and collates with training args.
    """
    if not results:
        print("Error: Results object is None.")
        return None, []

    # Initialize overall metrics
    map50_95, map50, map75, precision, recall, f1_score = None, None, None, None, None, None

    if hasattr(results, 'box'): 
        map50_95 = getattr(results.box, "map", None)
        map50 = getattr(results.box, "map50", None)
        map75 = getattr(results.box, "map75", None)
        
        if hasattr(results, 'mean_results'): 
            mean_res_values = results.mean_results()
            if len(mean_res_values) >= 2: (precision, recall) = mean_res_values[0:2]
            if len(mean_res_values) >= 3 and map50 is None: map50 = mean_res_values[2]
            if len(mean_res_values) >= 4 and map50_95 is None: map50_95 = mean_res_values[3]
        
        if precision is None and hasattr(results.box, 'p') and isinstance(results.box.p, (float, np.float_, int)):
            precision = results.box.p
        if recall is None and hasattr(results.box, 'r') and isinstance(results.box.r, (float, np.float_, int)):
            recall = results.box.r
        
        if precision is not None and recall is not None:
            if (precision + recall) > 0: f1_score = 2 * (precision * recall) / (precision + recall)
            else: f1_score = 0.0
            
    elif hasattr(results, 'metrics'): 
        print("Info: 'results.box' not found. Checking 'results.metrics'.")
        metrics_dict = results.metrics
        if metrics_dict:
            precision = metrics_dict.get('precision', precision)
            recall = metrics_dict.get('recall', recall)
            if precision is not None and recall is not None and f1_score is None:
                f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            if map50_95 is None: map50_95 = metrics_dict.get('mAP50-95', metrics_dict.get('map', None)) 
            if map50 is None: map50 = metrics_dict.get('mAP50', metrics_dict.get('map50', None))
            if map75 is None: map75 = metrics_dict.get('mAP75', metrics_dict.get('map75', None))
    else:
        print("Error: Results object lacks 'box' and 'metrics' attributes. Cannot extract detailed metrics.")
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
        "Precision_Overall": round(precision, 5) if precision is not None else 'N/A', # Clarified key
        "Recall_Overall": round(recall, 5) if recall is not None else 'N/A',       # Clarified key
        "F1_Score_Overall": round(f1_score, 5) if f1_score is not None else 'N/A',    # Clarified key
        "mAP50_Overall": round(map50, 5) if map50 is not None else 'N/A',           # Clarified key
        "mAP75_Overall": round(map75, 5) if map75 is not None else 'N/A',           # Clarified key
        "mAP50-95_Overall": round(map50_95, 5) if map50_95 is not None else 'N/A', # Clarified key
    }

    class_names_map = getattr(results, 'names', {})
    class_names_list = [class_names_map[i] for i in sorted(class_names_map.keys())] if class_names_map else []

    # --- Per-Class Metrics Extraction ---
    if hasattr(results, 'box'):
        per_class_p_array = getattr(results.box, 'p', None)      # Per-class Precision array
        per_class_r_array = getattr(results.box, 'r', None)      # Per-class Recall array
        per_class_f1_array = getattr(results.box, 'f1', None)     # Per-class F1-score array
        per_class_maps_array = getattr(results.box, 'maps', None) # Often per-class mAP50-95 for YOLO

        all_ap_values = getattr(results.box, 'all_ap', None) # (num_classes, num_iou_thresholds)

        for i, class_name in enumerate(class_names_list):
            safe_class_name = class_name.replace(' ', '_') # For consistent key naming

            if per_class_p_array is not None and i < len(per_class_p_array):
                collated_data[f"P_{safe_class_name}"] = round(per_class_p_array[i], 5)
            if per_class_r_array is not None and i < len(per_class_r_array):
                collated_data[f"R_{safe_class_name}"] = round(per_class_r_array[i], 5)
            if per_class_f1_array is not None and i < len(per_class_f1_array):
                collated_data[f"F1_{safe_class_name}"] = round(per_class_f1_array[i], 5)
            
            # Per-class mAP50 from all_ap (index 0 for IoU 0.5)
            if all_ap_values is not None and isinstance(all_ap_values, np.ndarray) and \
               all_ap_values.ndim == 2 and i < all_ap_values.shape[0] and all_ap_values.shape[1] > 0:
                collated_data[f"mAP50_{safe_class_name}"] = round(all_ap_values[i, 0], 5)
            
            # Per-class mAP75 from all_ap (index 5 for IoU 0.75 if IoUs are 0.5:0.05:0.95)
            if all_ap_values is not None and isinstance(all_ap_values, np.ndarray) and \
               all_ap_values.ndim == 2 and i < all_ap_values.shape[0] and all_ap_values.shape[1] > 5:
                collated_data[f"mAP75_{safe_class_name}"] = round(all_ap_values[i, 5], 5)

            # Per-class mAP50-95 (often from results.box.maps)
            if per_class_maps_array is not None and i < len(per_class_maps_array):
                collated_data[f"mAP50-95_{safe_class_name}"] = round(per_class_maps_array[i], 5)
    
    # --- Print Metrics ---
    print("\n--- Overall Metrics ---")
    print(f"  Precision (Overall): {collated_data.get('Precision_Overall', 'N/A')}")
    print(f"  Recall (Overall): {collated_data.get('Recall_Overall', 'N/A')}")
    print(f"  F1 Score (Overall): {collated_data.get('F1_Score_Overall', 'N/A')}")
    print(f"  mAP50 (Overall): {collated_data.get('mAP50_Overall', 'N/A')}")
    print(f"  mAP75 (Overall): {collated_data.get('mAP75_Overall', 'N/A')}")
    print(f"  mAP50-95 (Overall): {collated_data.get('mAP50-95_Overall', 'N/A')}")

    # Print Per-Class Metrics found
    # For brevity, we can print a summary here and rely on CSV for full details
    # Example for one per-class metric type:
    has_per_class = any(key.startswith("P_") for key in collated_data) # Check if any per-class P was added
    if has_per_class:
        print("\n--- Per-Class Metrics (Example: Precision) ---")
        for class_name in class_names_list:
            safe_class_name = class_name.replace(' ', '_')
            p_val = collated_data.get(f"P_{safe_class_name}", "N/A")
            if p_val != "N/A": # Only print if data exists
                 print(f"  P_{safe_class_name}: {p_val}")
        print("  (Full per-class P, R, F1, mAP50, mAP75, mAP50-95 saved to CSV)")


    return collated_data, class_names_list

# --- MAIN EXECUTION LOGIC ---
def perform_experiment_validations(
    input_dirs_list, data_yaml_path, output_base_dir, val_img_size, 
    val_conf_thresh, val_iou_thresh, device_setting, summary_csv_filename 
    ):
    os.makedirs(output_base_dir, exist_ok=True) 
    for single_input_dir in input_dirs_list:
        print(f"\nProcessing Model Directory: {single_input_dir}")
        print("=====================================================================")
        model_input_dir_name = os.path.basename(os.path.normpath(single_input_dir))
        model_path = os.path.join(single_input_dir, "weights", "best.pt")
        args_yaml_path = os.path.join(single_input_dir, "args.yaml")
        try:
            if not os.path.exists(model_path):
                print(f"ERROR: Model weights ... Skipping ...") # Abbreviated
                continue 
            training_params = load_training_args(args_yaml_path)
            ultralytics_project_for_val = output_base_dir 
            ultralytics_run_name_for_val = f"{model_input_dir_name}_val_c{val_conf_thresh}_i{val_iou_thresh}"
            specific_run_ultralytics_output_dir = os.path.join(ultralytics_project_for_val, ultralytics_run_name_for_val)
            print(f"Starting validation ... Ultralytics outputs will be in: {specific_run_ultralytics_output_dir}") # Abbreviated
            results = run_yolo_validation(
                model_path=model_path, data_yaml=data_yaml_path, imgsz_val=val_img_size, device=device_setting,
                project_dir=ultralytics_project_for_val, run_name=ultralytics_run_name_for_val,
                conf_thres=val_conf_thresh, iou_thres=val_iou_thresh,
            )
            if results:
                model_path_abs = os.path.abspath(model_path)
                data_yaml_abs = os.path.abspath(data_yaml_path)
                collated_metrics, class_names_list = process_and_collate_results(
                    results, training_params, model_input_dir_name, model_path_abs, data_yaml_abs, 
                    val_img_size, val_conf_thresh, val_iou_thresh
                )
                if collated_metrics:
                    os.makedirs(specific_run_ultralytics_output_dir, exist_ok=True) 
                    summary_csv_path_for_this_run = os.path.join(specific_run_ultralytics_output_dir, summary_csv_filename)
                    # THIS IS WHERE THE CALL HAPPENS
                    save_metrics_to_summary_csv(collated_metrics, summary_csv_path_for_this_run, class_names_list) 
                    print(f"Validation for '{model_input_dir_name}' complete.")
                else: print(f"Could not process or collate results for '{model_input_dir_name}'.")
            else: print(f"Validation run did not produce results for '{model_input_dir_name}'.")
        except FileNotFoundError as fnf_error: print(f"ERROR ... {fnf_error}") # Abbreviated
        except Exception as e:
            print(f"An unexpected error ... {model_input_dir_name}: {e}") # Abbreviated
            import traceback; traceback.print_exc()
        finally: print("---------------------------------------------------------------------")
    print(f"\nAll processed. Individual summaries in respective run folders inside '{output_base_dir}'.")

# --- SCRIPT ENTRY POINT & CONFIGURATION ---
if __name__ == "__main__":
    INPUT_DIRS_LIST = [ # Replace with your actual paths
        "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/Training_for_YOLO12m/Run1_YOLO12m_singeClass",
        
    ]
    DATA_YAML_PATH = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml" 
    #DATA_YAML_PATH = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf_multiple_Classes.yaml" # Verify this path
    OUTPUT_BASE_DIR_PATH = "./Valdation_resutls__rerun_yolo12m" # Make specific for this script
    SUMMARY_CSV_FILENAME = "run_metrics_summary_kv.csv" 
    VAL_IMAGE_SIZE = 640       
    VAL_CONF_THRESHOLD = 0.01 
    VAL_IOU_THRESHOLD = 0.7   
    DEVICE_SETTING = "cuda" if torch.cuda.is_available() else "cpu" 
    
    try: _ = pd.Timestamp.now(); _ = torch.cuda.is_available()
    except NameError: print("Import error..."); exit(1) # Abbreviated
    
    if not all(os.path.exists(p) for p in INPUT_DIRS_LIST if p) or not os.path.exists(DATA_YAML_PATH):
         print("\nWARNING: One or more configured INPUT_DIRS_LIST paths or DATA_YAML_PATH do not exist. Please verify.")
         # Decide if to exit or proceed; for now, it will try if list is not empty.
    
    if INPUT_DIRS_LIST and any(p for p in INPUT_DIRS_LIST): # Check if list is not empty and contains non-empty strings
        perform_experiment_validations(
            input_dirs_list=[p for p in INPUT_DIRS_LIST if p], # Filter out empty strings just in case
            data_yaml_path=DATA_YAML_PATH, output_base_dir=OUTPUT_BASE_DIR_PATH,
            val_img_size=VAL_IMAGE_SIZE, val_conf_thresh=VAL_CONF_THRESHOLD,
            val_iou_thresh=VAL_IOU_THRESHOLD, device_setting=DEVICE_SETTING,
            summary_csv_filename=SUMMARY_CSV_FILENAME
        )
    else: print("\nINFO: INPUT_DIRS_LIST is empty or contains only empty paths. No models to validate.")