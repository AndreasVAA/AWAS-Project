import os
import glob
import torch
import numpy as np
import csv
import time
from ultralytics import YOLO

# ----------------- Hardcoded Configuration -----------------
MODEL_PATH   = "/path/to/best.pt"        # update with your model path
DEVICE       = "cuda:0"                    # or "cpu"
IMG_SIZE     = 1280 # Consider TUPPLE (1280x960)
CONF_THRESH  = 0.4
IOU_THRESH   = 0.6
WARMUP_ITERS = 10

# Set your source folder and output CSV path directly
SOURCE_DIR   = "/path/to/images"         # update to your images folder
OUTPUT_CSV   = "yolo_full_timings.csv"

# ----------------- Full-Prediction Timer -----------------
def time_full_prediction(fn, *args):
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    outputs = fn(*args)

    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) * 1000.0, outputs  # ms

# ----------------- YOLO Prediction Wrapper -----------------
def predict_yolo(img_path):
    results = yolo.predict(
        source=img_path,
        imgsz=IMG_SIZE,
        device=DEVICE,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        save=False,
        verbose=False
    )
    return results

# ----------------- Main Benchmark Logic -----------------
def run_benchmark():
    # 1) Load YOLO model
    global yolo
    yolo = YOLO(MODEL_PATH)
    _ = yolo.model.to(DEVICE).eval()

    # 2) Gather image paths without a helper function
    patterns = ["*.jpg", "*.png", "*.jpeg"]
    img_paths = []
    for pat in patterns:
        img_paths += glob.glob(os.path.join(SOURCE_DIR, pat))
    img_paths = sorted(img_paths)
    if not img_paths:
        raise RuntimeError("No images found in SOURCE_DIR")

    # 3) Warm-up full pipeline on first image
    first = img_paths[0]
    print(f"Warm-up full-pipeline: {WARMUP_ITERS} runs on {first}")
    for _ in range(WARMUP_ITERS):
        _ = predict_yolo(first)

    # 4) Timing loop
    times_ms = []
    print(f"Timing full prediction over {len(img_paths)} images…")
    for img_path in img_paths:
        t_ms, _ = time_full_prediction(predict_yolo, img_path)
        times_ms.append(t_ms)

    # 5) Write results to CSV
    os.makedirs(os.path.dirname(OUTPUT_CSV) or ".", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "full_latency_ms"]);
        for p, t in zip(img_paths, times_ms):
            writer.writerow([p, f"{t:.3f}"])
        writer.writerow([])
        arr = np.array(times_ms)
        writer.writerow(["metric", "value"]);
        writer.writerow(["mean_ms", f"{arr.mean():.3f}"])
        writer.writerow(["std_ms",  f"{arr.std():.3f}"])
        writer.writerow(["p50_ms",  f"{np.percentile(arr,50):.3f}"])
        writer.writerow(["p90_ms",  f"{np.percentile(arr,90):.3f}"])
        writer.writerow(["p99_ms",  f"{np.percentile(arr,99):.3f}"])
        writer.writerow(["fps",     f"{1000.0/arr.mean():.2f}"])

    print(f"Done — full prediction results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()


"""
The timing captures every step from the moment an image file is handed off to the model wrapper until the final detection outputs are ready in memory. 
This includes reading the image from disk, performing all preprocessing operations (resizing, normalization, padding and tensor conversion), and copying the tensor onto the GPU. 
It then measures the full network forward pass through the YOLO layers, immediately followed by the non-max suppression filtering of raw detections. 
Finally, it records the time taken to assemble the filtered bounding boxes, scores and class labels into the returned results object.
"""