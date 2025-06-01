import os
import csv
import yaml
from ultralytics import YOLO # Use YOLO, not RTDETR for your YOLO script
import torch 
import pandas as pd
import numpy as np # Make sure numpy is imported if used by corrected function

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

def run_yolo_validation(model_path, data_yaml, imgsz_val, device, project_dir, run_name, conf_thres, iou_thres, **kwargs):
    """
    Run validation on the given data using the specified YOLO model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(data_yaml):
        raise FileNotFoundError(f"Data YAML file not found: {data_yaml}")

    model = YOLO(model_path) # Ensure this is YOLO for your YOLO script
    print(f"Running validation for {model_path} on {data_yaml} with imgsz={imgsz_val}, conf={conf_thres}, iou={iou_thres}")
    
    results = model.val(
        data=data_yaml,
        imgsz=imgsz_val,
        device=device,
        project=project_dir,
        name=run_name,      
        conf=conf_thres,    
        iou=iou_thres,      
        save_json=True,     
        save_hybrid=False,  # This was deprecated, but keeping as is for now if your ultralytics version handles it
        plots=True,         
        **kwargs
    )
    return results

def save_metrics_to_summary_csv_key_value_format(metrics_data, csv_filepath, class_names_list_for_ordering_if_needed): # class_names might be useful for ordering
    """
    Save metrics to a CSV file in a key-value pair format for a single run.
    """
    try:
        os.makedirs(os.path.dirname(csv_filepath), exist_ok=True)
        
        with open(csv_filepath, mode='w', newline='') as csv_file: # 'w' to overwrite for a single run's summary
            writer = csv.writer(csv_file)
            writer.writerow(["Metric", "Value"]) # Header for the key-value format

            # Write basic collated metrics first in a somewhat defined order
            # This order can be customized based on your preference for readability
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
            
            # Write per-class metrics if they exist
            if 'per_class_mAP50' in metrics_data and metrics_data['per_class_mAP50']:
                writer.writerow(["--- Per-Class mAP50 ---", "---"]) # Separator
                # Sort class names for consistent output order if desired
                sorted_per_class_keys = sorted(metrics_data['per_class_mAP50'].keys())
                for class_name in sorted_per_class_keys:
                    # Construct a descriptive metric name for the CSV
                    metric_name = f"mAP50_{class_name.replace(' ', '_')}" 
                    writer.writerow([metric_name, metrics_data['per_class_mAP50'].get(class_name, "N/A")])
            
            # Write any other metrics that might not have been in ordered_base_keys or per_class
            # This is a fallback to ensure all data is written
            writer.writerow(["--- Other Metrics ---", "---"]) # Separator
            existing_keys_written = set(ordered_base_keys)
            if 'per_class_mAP50' in metrics_data:
                for class_name in metrics_data['per_class_mAP50'].keys():
                    existing_keys_written.add(f"mAP50_{class_name.replace(' ', '_')}") # approximate key name

            for key, value in metrics_data.items():
                # Check if key approximates to something already written or is the dict itself
                if key not in ordered_base_keys and key != 'per_class_mAP50':
                     # A more robust check would be needed if keys have prefixes/suffixes
                    if key not in existing_keys_written: # Basic check
                        writer.writerow([key, value if value is not None else "N/A"])

        print(f"Metrics summary (key-value format) for this run saved to {csv_filepath}")

    except Exception as e:
        print(f"Error writing key-value CSV to {csv_filepath}: {e}")


# THIS IS THE CORRECTED FUNCTION TO USE IN YOUR YOLO SCRIPT
def process_and_collate_results(results, training_args, model_input_dir_name, model_path_abs, 
                                data_yaml_abs, val_imgsz, val_conf, val_iou):
    """
    Extracts metrics from YOLO results and collates with training args.
    Adapted based on common Ultralytics Results object structure (inspired by RT-DETR example for robustness).
    """
    if not results or not hasattr(results, 'box'):
        # Check if metrics are directly on results for some simpler cases or other model types if .box is not there
        if hasattr(results, 'metrics') and results.metrics is not None:
             print("Info: 'results.box' not found, but 'results.metrics' exists. Attempting to use 'results.metrics'.")
             box_metrics_dict = results.metrics # This is a dictionary for some result types
        elif hasattr(results, 'box') and not hasattr(results.box, 'metrics') and hasattr(results.box, 'map') :
             print("Info: 'results.box.metrics' not found, but 'results.box' attributes like 'map' exist. Attempting direct attribute access on results.box.")
             # We will use getattr(results.box, "map", None) etc. below
             pass # Allow proceeding to getattr checks
        else:
            print("Error: Results object is not as expected or lacks 'box' attribute, or 'box' lacks metric attributes, and no direct 'results.metrics'.")
            return None, []

    # --- Overall Detection Metrics ---
    map50_95 = getattr(results.box, "map", None) if hasattr(results, 'box') else None
    map50 = getattr(results.box, "map50", None) if hasattr(results, 'box') else None
    map75 = getattr(results.box, "map75", None) if hasattr(results, 'box') else None
    
    precision = None
    recall = None
    f1_score = None

    if hasattr(results, 'mean_results'): 
        mean_res_values = results.mean_results()
        if len(mean_res_values) >= 2:
            precision = mean_res_values[0]
            recall = mean_res_values[1]
            if precision is not None and recall is not None and (precision + recall) > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0
        if len(mean_res_values) >= 3 and map50 is None:
             map50 = mean_res_values[2]
        if len(mean_res_values) >= 4 and map50_95 is None:
             map50_95 = mean_res_values[3]
    
    # Fallback for P, R, F1 if not in mean_results and results.box exists
    if hasattr(results, 'box'):
        if precision is None and hasattr(results.box, 'p') and isinstance(results.box.p, (float, np.float_, int)): # Check for scalar
            precision = results.box.p
        if recall is None and hasattr(results.box, 'r') and isinstance(results.box.r, (float, np.float_, int)): # Check for scalar
            recall = results.box.r
        if f1_score is None and hasattr(results.box, 'f1') and isinstance(results.box.f1, (float, np.float_, int)): # Check for scalar
            f1_score = results.box.f1
        elif f1_score is None and precision is not None and recall is not None:
            if (precision + recall) > 0: f1_score = 2 * (precision * recall) / (precision + recall)
            else: f1_score = 0.0
    
    # If still no P, R, F1, and results.metrics was found (e.g. for classification)
    if precision is None and 'box_metrics_dict' in locals() and box_metrics_dict:
        # This part is for the case where results.metrics was used instead of results.box.metrics
        # You'd need to know the keys in box_metrics_dict if this path is taken.
        # Example generic keys (these are guesses, must be verified if this path is common for YOLO val):
        precision = box_metrics_dict.get('precision', precision) 
        recall = box_metrics_dict.get('recall', recall)
        # F1 might need calculation or also be in the dict
        if precision is not None and recall is not None and f1_score is None:
             if (precision + recall) > 0: f1_score = 2 * (precision * recall) / (precision + recall)
             else: f1_score = 0.0

        # Also try to get mAPs if they were in this results.metrics dict
        if map50_95 is None: map50_95 = box_metrics_dict.get('mAP50-95(B)', box_metrics_dict.get('map', None)) # Common keys
        if map50 is None: map50 = box_metrics_dict.get('mAP50(B)', box_metrics_dict.get('map50', None))
        if map75 is None: map75 = box_metrics_dict.get('mAP75(B)', box_metrics_dict.get('map75', None))


    collated_data = {
        "RunName": f"{model_input_dir_name}_val_c{val_conf}_i{val_iou}",
        "Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "ModelPath": model_path_abs, "DataYAML": data_yaml_abs,
        "TrainingImgSz": training_args.get('imgsz', 'N/A'), "ValidationImgSz": val_imgsz,
        "TrainingBatch": training_args.get('batch', 'N/A'), "TrainingOptimizer": training_args.get('optimizer', 'N/A'),
        "TrainingLR0": training_args.get('lr0', 'N/A'), "TrainingLRF": training_args.get('lrf', 'N/A'),
        "TrainingEpochs": training_args.get('epochs', 'N/A'), "TrainingModelDef": training_args.get('model', 'N/A'),
        "Val_Conf_Threshold": val_conf, "Val_IoU_Threshold": val_iou,
        "Precision": round(precision, 5) if precision is not None else 'N/A',
        "Recall": round(recall, 5) if recall is not None else 'N/A',
        "F1_Score": round(f1_score, 5) if f1_score is not None else 'N/A',
        "mAP50": round(map50, 5) if map50 is not None else 'N/A',
        "mAP75": round(map75, 5) if map75 is not None else 'N/A',
        "mAP50_95": round(map50_95, 5) if map50_95 is not None else 'N/A',
    }

    class_names_map = getattr(results, 'names', {})
    class_names_list = []
    if class_names_map:
        class_names_list = [class_names_map[i] for i in sorted(class_names_map.keys())]

    per_class_map50_dict = {}
    if hasattr(results, 'box') and hasattr(results.box, 'all_ap') and results.box.all_ap is not None:
        all_ap_values = results.box.all_ap 
        if all_ap_values.ndim == 2 and all_ap_values.shape[1] > 0: 
            for i, class_name in enumerate(class_names_list):
                if i < all_ap_values.shape[0]:
                    per_class_map50_dict[class_name] = round(all_ap_values[i, 0], 5) 
    
    if per_class_map50_dict:
        collated_data['per_class_mAP50'] = per_class_map50_dict
    
    print("\n--- Overall Metrics ---")
    print(f"  Precision (at max F1 or best available): {collated_data['Precision']}") # Clarified
    print(f"  Recall (at max F1 or best available): {collated_data['Recall']}")     # Clarified
    print(f"  F1 Score (max or best available): {collated_data['F1_Score']}")        # Clarified
    print(f"  mAP50: {collated_data['mAP50']}")
    print(f"  mAP75: {collated_data['mAP75']}")
    print(f"  mAP50-95: {collated_data['mAP50_95']}")

    if 'per_class_mAP50' in collated_data and collated_data['per_class_mAP50']:
        print("\n--- Per-Class mAP50 (from all_ap[:,0] if available) ---") 
        for class_name, pc_map50 in collated_data['per_class_mAP50'].items():
            print(f"  {class_name}: {pc_map50}")
    elif hasattr(results, 'box') and hasattr(results.box, 'maps') and results.box.maps is not None:
        print("\n--- Per-Class mAP (from results.box.maps, typically mAP50-95 per class) ---")
        # This is a fallback print if specific per-class mAP50 from all_ap wasn't populated
        # but results.box.maps exists. This data is NOT currently added to collated_data['per_class_mAP50']
        # as that key is specifically for mAP50. You could add another key for this if desired.
        for i, class_name in enumerate(class_names_list):
            if hasattr(results.box.maps, '__len__') and i < len(results.box.maps): # Check if maps is array-like
                 if isinstance(results.box.maps[i], (float, np.float_, int)):
                    print(f"  {class_name} (mAP via .maps): {round(results.box.maps[i], 5)}")
            elif isinstance(results.box.maps, (float, np.float_, int)) and i==0: # If maps is a scalar (e.g. single class)
                 print(f"  {class_name} (mAP via .maps): {round(results.box.maps, 5)}")


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
        model_path = os.path.join(single_input_dir, "weights", "best.pt")
        args_yaml_path = os.path.join(single_input_dir, "args.yaml")

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

            results = run_yolo_validation( # This function calls model.val()
                model_path=model_path,
                data_yaml=data_yaml_path,
                imgsz_val=val_img_size,
                device=device_setting,
                project_dir=ultralytics_project_for_val, 
                run_name=ultralytics_run_name_for_val, # Make sure this argument is passed correctly
                conf_thres=val_conf_thresh,
                iou_thres=val_iou_thresh,
            )

            if results:
                model_path_abs = os.path.abspath(model_path)
                data_yaml_abs = os.path.abspath(data_yaml_path)
                # Call the corrected process_and_collate_results
                collated_metrics, class_names_list = process_and_collate_results(
                    results, training_params, model_input_dir_name, 
                    model_path_abs, data_yaml_abs, val_img_size,
                    val_conf_thresh, val_iou_thresh
                )
                if collated_metrics:
                    os.makedirs(specific_run_ultralytics_output_dir, exist_ok=True) 
                    summary_csv_path_for_this_run = os.path.join(specific_run_ultralytics_output_dir, summary_csv_filename)
                    save_metrics_to_summary_csv(collated_metrics, summary_csv_path_for_this_run, class_names_list)
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
        # Example: "C:/Users/YourUser/Documents/YOLO_Runs/exp1_yolov8n_coco128",
        # Example: "/home/youruser/yolo_experiments/exp2_yolov8s_custom_data",
        "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/Testing_different_optimizers_640/YOLO11m_multiclass_640_batch4_Adam", 
        "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/Testing_different_optimizers_640/YOLO11m_multiclass_640_batch4_AdamW",
        "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/Testing_different_optimizers_640/YOLO11m_multiclass_640_batch4_SGD",
        "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/Testing_different_optimizers_640/YOLO11m_multiclass_640_batch4_RAdam",
    ]
    DATA_YAML_PATH = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml" #Swithc for mutiple classes
    OUTPUT_BASE_DIR_PATH = "./validation__optimizers_640_btch4" 
    SUMMARY_CSV_FILENAME = "val_metrics_summary.csv" 
    VAL_IMAGE_SIZE = 640       
    VAL_CONF_THRESHOLD = 0.001 
    VAL_IOU_THRESHOLD = 0.7    
    DEVICE_SETTING = "cuda" if torch.cuda.is_available() else "cpu" 
    # --- END OF CONFIGURABLE PARAMETERS ---

    try:
        _ = pd.Timestamp.now()
        _ = torch.cuda.is_available()
    except NameError: 
        print("Import error: Critical libraries (pandas, torch) might be missing or not imported correctly.")
        print("Please ensure PyYAML, pandas, ultralytics, and torch are installed.")
        exit(1)
    except Exception as e:
        print(f"An issue occurred with library checks: {e}")

    if not INPUT_DIRS_LIST or not DATA_YAML_PATH or \
       (len(INPUT_DIRS_LIST) > 0 and INPUT_DIRS_LIST[0].startswith("path/to/your")) or \
       DATA_YAML_PATH.startswith("path/to/your"):
        print("\nWARNING: Please update the placeholder paths in the 'CONFIGURABLE PARAMETERS' section of the script before running.")
    
    if INPUT_DIRS_LIST and not (len(INPUT_DIRS_LIST) == 1 and INPUT_DIRS_LIST[0].startswith("path/to/your")): # Check if not just the placeholder
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