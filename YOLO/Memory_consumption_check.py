import os
import torch
import pandas as pd
from pathlib import Path
from ultralytics import YOLO

def measure_memory_for_yolo_model(
    model_path, img_size=(640, 640), device="cuda"
):
    """
    Measure peak GPU memory usage during the full YOLO predict pipeline.
    """
    # 1) Initialize for detection and move to device
    model = YOLO(model_path, task="detect")
    model.model.to(device).eval()

    # 2) Create a dummy input
    H, W = img_size
    img = torch.randn(1, 3, H, W, device=device)

    # 3) Warm-up to compile kernels / allocate buffers
    with torch.no_grad():
        _ = model.predict(img, device=device, verbose=False)

    # 4) Clear leftover allocations
    torch.cuda.empty_cache()

    # 5) Reset and measure peak memory during a real inference
    torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        _ = model.predict(img, device=device, verbose=False)
        torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated(device)

    return peak_bytes / 1024**2  # MB

def process_directory(root_folder):
    """
    Walks through all subfolders of `root_folder`, finds `weights/best.pt`,
    measures each model’s peak inference memory, and writes a CSV.
    """
    results = []
    for subdir, dirs, files in os.walk(root_folder):
        if subdir == root_folder:
            continue

        best_pt = Path(subdir) / "weights" / "best.pt"
        if best_pt.exists():
            mem_mb = measure_memory_for_yolo_model(str(best_pt))
            results.append({
                "model_version": Path(subdir).name,
                "memory_usage_MB": mem_mb
            })

    df = pd.DataFrame(results)
    if df.empty:
        print("No results found.")
    else:
        out_name = f"{Path(root_folder).name}_memory_usage.csv"
        df.to_csv(out_name, index=False)
        print(f"Results saved to {out_name}")

if __name__ == "__main__":
    ROOT = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/New_new_Testing_batch_resolution_variations_single_class"
    process_directory(ROOT)
