from ultralytics import YOLO
import os

def train_model(
    # --- Core settings that usually need to be specified per run ---
    model_path: str,                # Path to the model file (e.g., "yolo11m.pt", "yolov8n.pt")
    config_path: str,               # Path to data.yaml
    project: str,                   # Project name for organizing experiments
    name: str,                      # Specific name for this run
    
    # --- Common training parameters with sensible defaults ---
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,                # Ultralytics default is often 16; can be overridden
    optimizer: str = 'auto',        # Ultralytics 'auto' picks based on context
    
    rect: bool = False,             # Rectangular training
    freeze: int = None,             # Freeze first N layers
    device: any = 0,                # CUDA device, e.g., 0, '0', '0,1,2,3', or 'cpu'
    workers: int = 8,               # Number of worker threads for data loading
    exist_ok: bool = False,         # False to prevent accidental overwrites
    seed: int = 0,                  # Random seed
    pretrained: bool = True,        # Start from pretrained model
    amp: bool = True,               # Automatic Mixed Precision
    patience: int = 75,             # Early stopping patience (user's previous preference)
    save: bool = True,              # Save checkpoints
    save_period: int = -1,          # Save checkpoint every X epochs
    verbose: bool = True,           # Verbose output
    deterministic: bool = True,     # For reproducibility
    single_cls: bool = False,       # Single-class training
    val: bool = True,               # Validate during training
    split: str = "val",             # Validation split
    
    # --- All other Ultralytics train() arguments (hyperparameters, specific controls) via kwargs ---
    **other_yolo_train_args
):
    """
    Trains a YOLO model using Ultralytics.
    - Core parameters (model_path, config_path, project, name) are required.
    - Common training parameters have defaults and can be overridden.
    - Any other keyword arguments provided in `**other_yolo_train_args` will be passed directly
      to the Ultralytics `model.train()` method. If a hyperparameter (e.g., lr0, mosaic)
      is NOT provided in the call to this function, it will NOT be passed to `model.train()`,
      allowing Ultralytics to use its own internal default for that parameter.
    """
    print(f"--- Training Run: {project}/{name} ---")
    print(f"Working directory: {os.getcwd()}")
    print(f"Model: {model_path}, Data: {config_path}")
    print(f"Settings: Epochs={epochs}, ImgSz={imgsz}, Batch={batch}, Optimizer={optimizer}")

    yolo_instance = YOLO(model_path, task="detect")

    # Start with parameters explicitly defined in this function's signature
    train_params_to_pass = {
        'data': config_path, 'project': project, 'name': name, 'epochs': epochs,
        'imgsz': imgsz, 'batch': batch, 'optimizer': optimizer, 'rect': rect,
        'freeze': freeze, 'device': device, 'workers': workers, 'exist_ok': exist_ok,
        'seed': seed, 'pretrained': pretrained, 'amp': amp, 'patience': patience,
        'save': save, 'save_period': save_period, 'verbose': verbose,
        'deterministic': deterministic, 'single_cls': single_cls, 'val': val, 'split': split
    }

    # Add all other arguments passed via **other_yolo_train_args
    # These will directly override any Ultralytics defaults if their names match.
    # If a hyperparameter key (e.g., 'lr0') is not in other_yolo_train_args,
    # it's not added here, so Ultralytics uses its internal default.
    if other_yolo_train_args:
        print("Applying additional/hyperparameter arguments from kwargs:")
        for k, v_ in other_yolo_train_args.items():
            print(f"  {k}: {v_}")
        train_params_to_pass.update(other_yolo_train_args)
    else:
        print("No additional kwargs for hyperparameters provided; Ultralytics defaults will apply for unspecified ones.")
    
    results = yolo_instance.train(**train_params_to_pass)
    return results


if __name__ == "__main__":
    # Define your specific tuned hyperparameters from the tuning run
    user_tuned_hyperparameters = {
        'lr0': 0.00038, 'lrf': 0.00939, 'momentum': 0.75246, 'weight_decay': 0.00083,
        'warmup_epochs': 2.77389, 'warmup_momentum': 0.94291,
        'box': 8.2063, 'cls': 0.84574, 'dfl': 0.75034,
        'hsv_h': 0.0256, 'hsv_s': 0.27606, 'hsv_v': 0.19502,
        'degrees': 1.2906, 'translate': 0.11209, 'scale': 0.36401, 'shear': 1.94656,
        'perspective': 0.00037, 'flipud': 0.24353, 'fliplr': 0.59228,
        'bgr': 0.0542, 'mosaic': 0.44041, 'mixup': 0.03207, 'copy_paste': 0.0651,
    }

    #data_config_file = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml" # Verify this path
    data_config_file = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf_multiple_Classes.yaml"

    run_configurations = [
        {
            # Run 1: Uses YOUR specific tuned hyperparameters
            "model_path": "yolo11m.pt",
            "config_path": data_config_file,
            "project": "Training_for_full_tuned_hyperparamters_multiclass", # Changed project name slightly for clarity
            "name": "Run1_YOLO11m_UserTuned_multiclass",
            "imgsz": 640,
            "batch": 4,
            "optimizer": 'AdamW', # This was part of your tuned setup
            "epochs": 500,
            **user_tuned_hyperparameters # These will be caught by **other_yolo_train_args
        },
    ]

    for i, current_run_config in enumerate(run_configurations):
        print(f"\n>>> Processing Run {i+1} of {len(run_configurations)}: {current_run_config.get('project')}/{current_run_config.get('name')} <<<")
        try:
            train_results = train_model(**current_run_config)
            if hasattr(train_results, 'save_dir'):
                 print(f"Run '{current_run_config['name']}' completed. Results saved to: {train_results.save_dir}")
            else:
                 print(f"Run '{current_run_config['name']}' completed, but no save_dir attribute found in results object.")

        except Exception as e:
            print(f"!!! Error during run '{current_run_config.get('name', 'UnnamedRun')}': {e}")
            import traceback
            traceback.print_exc()
            print(f"Problematic configuration: {current_run_config}")

    print("\n--- All configured training runs attempted. ---")