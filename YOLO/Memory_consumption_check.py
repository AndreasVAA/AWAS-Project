import os
import torch
import pandas as pd
import subprocess
from pathlib import Path
from ultralytics import YOLO 

def get_memory_usage():
    """
    Function to get the current GPU memory usage using nvidia-smi.
    Returns the memory usage in MB.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True
        )
        memory_used = int(result.stdout.strip())  # memory used in MB
        return memory_used
    except Exception as e:
        print(f"Error getting GPU memory usage: {e}")
        return None



def measure_memory_for_yolo_model(
    model_path, img_size=(640,640), device="cuda"
):
    # load & warm model
    model = YOLO(model_path).model.to(device).eval()
    # dummy input
    H,W = img_size
    x = torch.randn(1,3,H,W, device=device)

    # measure peak memory
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        model(x)
        torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)
    return peak_bytes / 1024**2  # MB

    return mem_after - mem_before if mem_before is not None else None

def process_directory(root_folder):
    """
    Processes the root folder and its subfolders, measuring memory consumption for each YOLO model.
    """
    results = []

    # Walk through all subfolders in the root directory
    for subdir, dirs, files in os.walk(root_folder):
        # Skip the root folder itself and non-YOLO directories
        if subdir == root_folder:
            continue

        # Check if this subfolder has a 'weights' folder and 'best.pt' file
        weights_folder = Path(subdir) / 'weights'
        best_pt = weights_folder / 'best.pt'
        
        if best_pt.exists():
            # Measure memory usage for the model
            memory_usage = measure_memory_for_yolo_model(str(best_pt))
            
            # Use the subfolder name (not the full path) for model version identification
            model_version = Path(subdir).name

            # Save the results in a dictionary
            if memory_usage is not None:
                results.append({
                    'model_version': model_version,
                    'memory_usage_MB': memory_usage
                })

    # Create a DataFrame and save to CSV
    df = pd.DataFrame(results)
    if df.empty:
        print("No results found.")
    else:
        # Save the output CSV with the root folder's name
        output_file = f"{os.path.basename(root_folder)}_memory_usage.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

# Usage example
root_folder = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/New_new_Testing_batch_resolution_variations_single_class"  # Replace this with the actual folder path
process_directory(root_folder)
