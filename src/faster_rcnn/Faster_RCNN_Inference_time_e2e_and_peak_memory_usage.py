import os
import glob
import torch
import numpy as np
import csv
import time
import cv2 # For image loading and preprocessing
import torchvision.transforms.functional as TF # For tensor conversion
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
import torchvision.models.detection.faster_rcnn # For FastRCNNPredictor

# ----------------- Configuration (Update these) -----------------
# --- Model & Paths ---
FASTER_RCNN_MODEL_PATH = "/home/itk/Desktop/Andreas/AWAS-Project/FasterR_CNN/FasterRCNN_Runs_Third_iteraion/SGD_SingleClass_640_NoAccum_F1Opt_LR0.005_BS4_ImgSz640/best_model.pth" # IMPORTANT: Update with your Faster R-CNN model path
CLASS_NAMES = ["Plankton"] # For single-class, or your multi-class list
                             # e.g., ["Tripos longipes", "Tripos fusus", ...]
SOURCE_DIR = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/images" # Update to your images folder
OUTPUT_CSV = "faster_rcnn_benchmark_results.csv" # Name of the output CSV

# --- Device & Inference Parameters ---
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (640, 640) # Target image size for preprocessing (height, width)
                       # Should match what the model was trained or is intended to be used with.
                       # The YOLO script had (1024,1024) so using this for consistency.
                       # Adjust if your Faster R-CNN model expects a different size (e.g., 640,640).

# --- Benchmarking Parameters ---
BATCH_MODE = True    # True for batch inference, False for single-image
BATCH_SIZE = 16      # Batch size for inference (user requested 16)
WARMUP_ITERS = 10    # Number of warm-up iterations

# Note: CONF_THRESH and IOU_THRESH from the YOLO script affect its internal NMS.
# For torchvision's Faster R-CNN, NMS parameters are typically part of the model's construction
# or its forward pass internal logic. We are benchmarking the model as is.
# If you need to apply additional filtering *after* getting results and time that too,
# it would be an extra step. For this script, we focus on the model's output generation speed.

# ----------------- Helper: Model Definition (from your script) -----------------
def get_faster_rcnn_model(num_classes_with_background):
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes_with_background)
    return model

# ----------------- Helper: Image Preprocessing -----------------
def load_and_preprocess_image(image_path, target_size):
    """Loads an image, resizes, converts color, and transforms to tensor."""
    img = cv2.imread(image_path)
    if img is None:
        print(f"Warning: Failed to load image {image_path}. Skipping.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_LINEAR) # (width, height) for cv2.resize
    img_tensor = TF.to_tensor(img_resized)
    return img_tensor

# ----------------- Full-Prediction + GPU-Mem Timer (from YOLO script) -----------------
def time_full_prediction(prediction_fn, batch_image_paths_or_single_path):
    """
    Times the execution of prediction_fn and measures peak GPU memory.
    prediction_fn should handle everything from image path(s) to model output.
    """
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize(device=DEVICE) # Ensure previous ops are done
        torch.cuda.reset_peak_memory_stats(device=DEVICE)
    
    t0 = time.perf_counter()
    # The prediction_fn is responsible for all steps: loading, preprocessing, model forward
    outputs = prediction_fn(batch_image_paths_or_single_path)
    
    if DEVICE.startswith("cuda"):
        torch.cuda.synchronize(device=DEVICE) # Ensure prediction ops are done
        peak_bytes = torch.cuda.max_memory_allocated(device=DEVICE)
    else:
        peak_bytes = 0 # No GPU memory to track on CPU
        
    t1 = time.perf_counter()

    latency_ms = (t1 - t0) * 1000.0
    peak_mb = peak_bytes / (1024**2)
    return latency_ms, peak_mb, outputs

# ----------------- Faster R-CNN Prediction Wrapper -----------------
# This global model variable will be initialized in run_benchmark
faster_rcnn_model = None

def predict_faster_rcnn(batch_image_paths): # Expects a list of paths for a batch
    """
    Handles image loading, preprocessing, and prediction for a batch of image paths.
    This entire function's execution time is measured.
    """
    global faster_rcnn_model # Use the globally loaded model
    
    images_processed = []
    for img_path in batch_image_paths:
        img_tensor = load_and_preprocess_image(img_path, IMG_SIZE)
        if img_tensor is not None:
            images_processed.append(img_tensor)
    
    if not images_processed:
        return [] # Return empty if no images could be processed

    # Batch images and move to device
    # Note: torchvision models expect a list of tensors for variable sized inputs,
    # but for fixed size inputs common in benchmarking, stacking can be done.
    # However, Faster R-CNN in training mode and often in eval mode takes a list of image tensors.
    image_batch_on_device = [img.to(DEVICE) for img in images_processed]

    with torch.no_grad():
        predictions = faster_rcnn_model(image_batch_on_device)
    
    return predictions


# ----------------- Main Benchmark Logic -----------------
def run_benchmark():
    global faster_rcnn_model # To assign the loaded model

    print(f"Using device: {DEVICE}")
    print(f"Loading Faster R-CNN model from: {FASTER_RCNN_MODEL_PATH}")
    num_classes = len(CLASS_NAMES)
    num_model_classes = num_classes + 1 # Add 1 for the background class
    
    faster_rcnn_model = get_faster_rcnn_model(num_model_classes)
    try:
        checkpoint = torch.load(FASTER_RCNN_MODEL_PATH, map_location=torch.device('cpu')) # Load to CPU first
        # Adjust key names if necessary (e.g. if saved from DataParallel or DistributedDataParallel)
        # For typical torchvision saves, direct loading should work.
        # If 'state_dict' was used: checkpoint = checkpoint['state_dict']
        faster_rcnn_model.load_state_dict(checkpoint)
        print("Model weights loaded successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        print("Ensure FASTER_RCNN_MODEL_PATH is correct and the checkpoint is compatible.")
        return

    faster_rcnn_model.to(DEVICE).eval()
    print(f"Model moved to {DEVICE} and set to evaluation mode.")

    # Gather image paths
    img_extensions = ["*.jpg", "*.png", "*.jpeg"]
    img_paths = []
    for ext in img_extensions:
        img_paths.extend(glob.glob(os.path.join(SOURCE_DIR, ext)))
    img_paths = sorted(img_paths)

    if not img_paths:
        print(f"Error: No images found in SOURCE_DIR: {SOURCE_DIR} with extensions {img_extensions}")
        return
    print(f"Found {len(img_paths)} images for benchmarking in {SOURCE_DIR}.")

    # Warm-up
    print(f"Starting warm-up ({WARMUP_ITERS} iterations)...")
    num_warmup_samples = BATCH_SIZE if BATCH_MODE else 1
    if not img_paths: # Should have been caught earlier, but as a safeguard
        print("Error: No images available for warm-up.")
        return

    warmup_inputs = img_paths[:num_warmup_samples]
    if len(warmup_inputs) < num_warmup_samples and len(img_paths) > 0: # handle case with fewer images than batch_size
        warmup_inputs = [img_paths[0]] * num_warmup_samples # duplicate first image if not enough unique
        if not BATCH_MODE: warmup_inputs = warmup_inputs[0] # if single mode, just one path

    if not warmup_inputs:
        print("Error: Not enough images for warm-up, even after trying to duplicate.")
        return

    for i in range(WARMUP_ITERS):
        print(f"Warm-up iteration {i+1}/{WARMUP_ITERS}")
        # predict_faster_rcnn expects a list, even for a single image if BATCH_MODE is True conceptually
        # but for clarity, we'll match how it's called in the main loop.
        current_warmup_batch = warmup_inputs if isinstance(warmup_inputs, list) else [warmup_inputs]
        _ = predict_faster_rcnn(current_warmup_batch)
    print("Warm-up complete.")

    # Timing + memory measurements
    results_data = []  # list of (run_id_or_image_index, latency_ms, peak_gpu_mem_mb, num_images_in_batch)
    
    print(f"Starting benchmark with BATCH_MODE: {BATCH_MODE}, BATCH_SIZE: {BATCH_SIZE if BATCH_MODE else 1}")

    if BATCH_MODE:
        for idx in range(0, len(img_paths), BATCH_SIZE):
            batch_paths = img_paths[idx : idx + BATCH_SIZE]
            if not batch_paths: continue # Should not happen if len(img_paths) > 0

            print(f"Processing batch {idx // BATCH_SIZE + 1}/{(len(img_paths) + BATCH_SIZE - 1) // BATCH_SIZE} (images {idx+1}-{idx+len(batch_paths)})")
            
            # time_full_prediction itself calls predict_faster_rcnn
            latency_ms, peak_mb, _ = time_full_prediction(predict_faster_rcnn, batch_paths)
            
            run_id = idx // BATCH_SIZE
            results_data.append((run_id, latency_ms, peak_mb, len(batch_paths)))
    else: # Single image mode
        for idx, img_path in enumerate(img_paths):
            print(f"Processing image {idx + 1}/{len(img_paths)}: {os.path.basename(img_path)}")
            
            # predict_faster_rcnn expects a list of paths, so wrap single path in a list
            latency_ms, peak_mb, _ = time_full_prediction(predict_faster_rcnn, [img_path])
            
            results_data.append((idx, latency_ms, peak_mb, 1)) # image_index, latency, memory, num_images=1

    if not results_data:
        print("No results were generated. Check image paths and processing logic.")
        return

    # Write results to CSV
    output_dir = os.path.dirname(OUTPUT_CSV)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["run_id", "latency_ms_total_batch", "peak_gpu_mem_mb", "num_images_in_batch"])
        for run_id, lat_ms, peak_mb_val, num_imgs in results_data:
            writer.writerow([run_id, f"{lat_ms:.3f}", f"{peak_mb_val:.3f}", num_imgs])

        # Summary statistics
        writer.writerow([])
        writer.writerow(["metric", "value"])
        
        # Latencies are per batch. For mean latency per image, we need to divide by num_images_in_batch
        # However, for FPS calculation, total time / total images is more direct.
        total_time_ms = sum(r[1] for r in results_data)
        total_images_processed = sum(r[3] for r in results_data)
        
        mean_latency_ms_per_batch = total_time_ms / len(results_data) if results_data else 0
        
        # For per-image latency stats, we need individual per-image latencies
        # This requires a slight adjustment if BATCH_MODE is True, as lat_ms is for the whole batch
        per_image_latencies_ms = []
        for _, lat_ms, _, num_imgs_in_batch_val in results_data:
            if num_imgs_in_batch_val > 0:
                 per_image_latencies_ms.extend([lat_ms / num_imgs_in_batch_val] * num_imgs_in_batch_val)
            
        if per_image_latencies_ms:
            per_image_latencies_np = np.array(per_image_latencies_ms)
            writer.writerow(["mean_latency_ms_per_image", f"{per_image_latencies_np.mean():.3f}"])
            writer.writerow(["p50_latency_ms_per_image", f"{np.percentile(per_image_latencies_np, 50):.3f}"])
            writer.writerow(["p90_latency_ms_per_image", f"{np.percentile(per_image_latencies_np, 90):.3f}"])
        else:
            writer.writerow(["mean_latency_ms_per_image", "N/A"])
            writer.writerow(["p50_latency_ms_per_image", "N/A"])
            writer.writerow(["p90_latency_ms_per_image", "N/A"])

        mems_mb = np.array([r[2] for r in results_data if r[2] > 0]) # Filter out CPU runs if any for mem stats
        if len(mems_mb) > 0:
            writer.writerow(["max_peak_gpu_mem_mb", f"{mems_mb.max():.3f}"])
        else:
            writer.writerow(["max_peak_gpu_mem_mb", "N/A (No GPU usage recorded)"])

        # Throughput (FPS)
        if total_time_ms > 0:
            fps = total_images_processed * 1000.0 / total_time_ms
            writer.writerow(["fps_images_per_sec", f"{fps:.2f}"])
        else:
            writer.writerow(["fps_images_per_sec", "N/A"])
            
        writer.writerow(["mean_latency_ms_per_batch", f"{mean_latency_ms_per_batch:.3f}"])


    print(f"Benchmark complete. Results saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_benchmark()