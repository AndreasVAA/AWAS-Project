import os
import glob
import torch
import numpy as np
import csv
import time
from ultralytics import YOLO

# ----------------- Hardcoded Configuration -----------------
MODEL_PATH   = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/Training_for_YOLO12m/Run1_YOLO12m_singeClass/weights/best.pt"  # update with your model path
DEVICE       = "cuda:0"  # or "cpu"
IMG_SIZE     = 640
CONF_THRESH  = 0.01
IOU_THRESH   = 0.7
WARMUP_ITERS = 10

# Toggle between single-image and batch processing
BATCH_MODE   = True  # True for batch inference, False for single-image
BATCH_SIZE   = 16     # valid when BATCH_MODE is True

SOURCE_DIR   = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/images"  # update to your images folder
OUTPUT_CSV   = "yolo12m_640_benchmark_results.csv"

# ----------------- Full-Prediction + GPU-Mem Timer -----------------
def time_full_prediction(fn, *args):
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device=DEVICE)
    t0 = time.perf_counter()
    outputs = fn(*args)
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
        peak_bytes = torch.cuda.max_memory_allocated(device=DEVICE)
    else:
        peak_bytes = 0
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0
    peak_mb    = peak_bytes / (1024**2)
    return latency_ms, peak_mb, outputs  # ms, MB

# ----------------- YOLO Prediction Wrapper -----------------
def predict_yolo(inputs):
    """
    inputs: single image path (str) or list of paths
    """
    return yolo.predict(
        source=inputs,
        batch=BATCH_SIZE if BATCH_MODE else 1,
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=False,
        verbose=False
    )

# ----------------- Main Benchmark Logic -----------------
def run_benchmark():
    global yolo
    yolo = YOLO(MODEL_PATH)
    _ = yolo.model.to(DEVICE).eval()

    # gather image paths
    img_paths = sorted(
        glob.glob(os.path.join(SOURCE_DIR, "*.jpg")) +
        glob.glob(os.path.join(SOURCE_DIR, "*.png")) +
        glob.glob(os.path.join(SOURCE_DIR, "*.jpeg"))
    )
    if not img_paths:
        raise RuntimeError("No images found in SOURCE_DIR")

    # Warm-up
    warmup_inputs = img_paths[:BATCH_SIZE] if BATCH_MODE else img_paths[0]
    for _ in range(WARMUP_ITERS):
        _ = predict_yolo(warmup_inputs)

    # Timing + memory measurements
    results = []  # list of (run_id, latency_ms, peak_gpu_mem_mb)
    if BATCH_MODE:
        for idx in range(0, len(img_paths), BATCH_SIZE):
            batch = img_paths[idx: idx + BATCH_SIZE]
            t_ms, mem_mb, _ = time_full_prediction(predict_yolo, batch)
            run_id = idx // BATCH_SIZE
            results.append((run_id, t_ms, mem_mb))
    else:
        for idx, img_path in enumerate(img_paths):
            t_ms, mem_mb, _ = time_full_prediction(predict_yolo, img_path)
            results.append((idx, t_ms, mem_mb))

    # Write results to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "latency_ms", "peak_gpu_mem_mb"])
        for run_id, t, m in results:
            w.writerow([run_id, f"{t:.3f}", f"{m:.3f}"])

        # summary stats
        w.writerow([])
        w.writerow(["metric", "value"])
        times = np.array([r[1] for r in results])
        mems  = np.array([r[2] for r in results])
        # latency metrics
        w.writerow(["mean_latency_ms",     f"{times.mean():.3f}"])
        w.writerow(["p50_latency_ms",      f"{np.percentile(times, 50):.3f}"])
        w.writerow(["p90_latency_ms",      f"{np.percentile(times, 90):.3f}"])
        # memory metric (peak requirement)
        w.writerow(["max_peak_gpu_mem_mb", f"{mems.max():.3f}"])
        # throughput
        images_per_batch = BATCH_SIZE if BATCH_MODE else 1
        fps = images_per_batch * 1000.0 / times.mean()
        w.writerow(["fps_images_per_sec",   f"{fps:.2f}"])

    print(f"Done — results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()
