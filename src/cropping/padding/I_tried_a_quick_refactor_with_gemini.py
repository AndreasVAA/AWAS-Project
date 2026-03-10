import cv2
import numpy as np
import os
import glob
from typing import Tuple, List, Dict, Optional, Any

# =============================================================================
# Cropping Pipeline Configuration Notes:
#
# - Image Resizing: Input images are first resized to a `global_target_size`.
# - Annotation Format: Expects YOLO-style annotations (class_id, x_c, y_c, w, h - normalized).
#   These are converted to normalized corner coordinates (xmin, ymin, xmax, ymax).
# - Padding Strategy for Crops:
#   1. Fixed Padding: A margin, defined by `fixed_pad_ratio` relative to the
#      object's maximum dimension (S = max(width, height)), is always added to
#      preserve context. (e.g., 12.5% of S).
#   2. Dynamic Padding: If the fixed-padded region is smaller than the
#      `classification_target_size` (e.g., 224x224), additional padding is added
#      to meet this minimum size. The final crop is not resized further after this.
#   3. Background Filling: If padding extends beyond original image boundaries,
#      the new area is filled using a color derived from the original image,
#      controlled by `fill_mode` (e.g., 'mean', 'median', 'gaussian').
# =============================================================================

def compute_background(image: np.ndarray, fill_mode: str = 'mean') -> Tuple[int, int, int]:
    """
    Compute a background fill value from the full image.

    Args:
        image: The input image (NumPy array).
        fill_mode: Method to compute background ('mean', 'median', 'gaussian').
                   Defaults to 'mean'.

    Returns:
        A BGR color tuple (e.g., (128, 128, 128)).
    """
    if fill_mode == 'mean':
        bg_color_float = np.mean(image, axis=(0, 1))
    elif fill_mode == 'median':
        bg_color_float = np.median(image, axis=(0, 1))
    elif fill_mode == 'gaussian':
        blurred = cv2.GaussianBlur(image, (31, 31), 0)
        # Use the center pixel of the heavily blurred image
        center_row = blurred.shape[0] // 2
        center_col = blurred.shape[1] // 2
        bg_color_float = blurred[center_row, center_col]
    else:
        # Default to black if mode is unrecognized
        bg_color_float = np.array([0, 0, 0])
    return tuple(map(int, bg_color_float))

def _resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    Resize the image to target_size (width, height).

    Args:
        image: The input image (NumPy array).
        target_size: A tuple (width, height) for resizing.

    Returns:
        The resized image (NumPy array).
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

def parse_label_file(label_path: str) -> List[Dict[str, Any]]:
    """
    Parse a YOLO-format label file.

    Each line should contain 5 values: class_id, x_center, y_center, width, height
    or 6 values (the sixth being confidence). All coordinates are normalized.
    Converts center-based annotations to normalized corner-based coordinates (xmin, ymin, xmax, ymax).

    Args:
        label_path: Path to the YOLO label file.

    Returns:
        A list of detection dictionaries, each with 'class_id', 'bbox' (normalized
        corner coordinates), and optionally 'conf'.
    """
    detections: List[Dict[str, Any]] = []
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} not found.")
        return detections
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) not in [5, 6]:
                print(f"Skipping invalid annotation line: {line.strip()}")
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center = float(parts[1]), float(parts[2])
                width_norm, height_norm = float(parts[3]), float(parts[4])
            except ValueError:
                print(f"Error parsing coordinate values in line: {line.strip()}")
                continue

            x_min = x_center - width_norm / 2
            y_min = y_center - height_norm / 2
            x_max = x_center + width_norm / 2
            y_max = y_center + height_norm / 2
            detection: Dict[str, Any] = {'class_id': class_id, 'bbox': (x_min, y_min, x_max, y_max)}

            if len(parts) == 6:
                try:
                    detection['conf'] = float(parts[5])
                except ValueError:
                    print(f"Error parsing confidence in line: {line.strip()}, setting to None.")
                    detection['conf'] = None
            detections.append(detection)
    return detections

def crop_with_padding(
    image: np.ndarray,
    norm_bbox: Tuple[float, float, float, float],
    fixed_pad_ratio: float = 0.125,
    classification_target_size: Tuple[int, int] = (224, 224),
    fill_mode: str = 'mean'
) -> np.ndarray:
    """
    Crop an instance from the image using a normalized (corner) bbox with padding.

    The process involves:
    1. Convert normalized bbox to absolute coordinates on the given image.
    2. Apply fixed padding (proportional to object size) to all sides.
    3. Apply dynamic padding if the fixed-padded region is smaller than
       `classification_target_size`, to meet this minimum size.
    4. Fill any areas outside the original image boundaries using `fill_mode`.

    Args:
        image: The image (NumPy array) from which to crop.
        norm_bbox: Normalized bounding box (xmin, ymin, xmax, ymax).
        fixed_pad_ratio: Ratio for fixed padding (e.g., 0.125 for 12.5%).
        classification_target_size: Tuple (width, height) for minimum crop size.
        fill_mode: Method for background filling ('mean', 'median', 'gaussian').

    Returns:
        The final cropped image (NumPy array), guaranteed to be at least
        `classification_target_size`.
    """
    h, w = image.shape[:2]
    xmin_abs = int(norm_bbox[0] * w)
    ymin_abs = int(norm_bbox[1] * h)
    xmax_abs = int(norm_bbox[2] * w)
    ymax_abs = int(norm_bbox[3] * h)

    box_w = xmax_abs - xmin_abs
    box_h = ymax_abs - ymin_abs
    S = max(box_w, box_h) # Max dimension of the original bbox

    # 1. Apply fixed padding
    fixed_pad_pixels = int(S * fixed_pad_ratio)
    padded_xmin = xmin_abs - fixed_pad_pixels
    padded_ymin = ymin_abs - fixed_pad_pixels
    padded_xmax = xmax_abs + fixed_pad_pixels
    padded_ymax = ymax_abs + fixed_pad_pixels

    current_width = padded_xmax - padded_xmin
    current_height = padded_ymax - padded_ymin
    target_w, target_h = classification_target_size

    # 2. Apply dynamic padding if needed
    dynamic_pad_left = dynamic_pad_right = dynamic_pad_top = dynamic_pad_bottom = 0
    if current_width < target_w:
        extra_pad_x = target_w - current_width
        dynamic_pad_left = extra_pad_x // 2
        dynamic_pad_right = extra_pad_x - dynamic_pad_left
    if current_height < target_h:
        extra_pad_y = target_h - current_height
        dynamic_pad_top = extra_pad_y // 2
        dynamic_pad_bottom = extra_pad_y - dynamic_pad_top

    final_xmin = padded_xmin - dynamic_pad_left
    final_ymin = padded_ymin - dynamic_pad_top
    final_xmax = padded_xmax + dynamic_pad_right
    final_ymax = padded_ymax + dynamic_pad_bottom

    final_crop_width = final_xmax - final_xmin
    final_crop_height = final_ymax - final_ymin

    # 3. Create crop canvas and fill
    # Check if the entire crop is within image boundaries
    if final_xmin >= 0 and final_ymin >= 0 and final_xmax <= w and final_ymax <= h:
        crop = image[final_ymin:final_ymax, final_xmin:final_xmax].copy()
    else:
        # Create a new canvas with background color
        bg_color = compute_background(image, fill_mode=fill_mode)
        crop = np.full((final_crop_height, final_crop_width, image.shape[2] if image.ndim == 3 else 1),
                       bg_color, dtype=image.dtype)

        # Define source (from original image) and destination (on crop canvas) coordinates
        src_x1 = max(final_xmin, 0)
        src_y1 = max(final_ymin, 0)
        src_x2 = min(final_xmax, w)
        src_y2 = min(final_ymax, h)

        dst_x1 = src_x1 - final_xmin
        dst_y1 = src_y1 - final_ymin
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)

        # Copy the valid region from the image to the crop canvas
        if src_x2 > src_x1 and src_y2 > src_y1 : # Ensure there's a valid area to copy
             crop[dst_y1:dst_y2, dst_x1:dst_x2] = image[src_y1:src_y2, src_x1:src_x2]
    return crop

def create_crops_from_image_and_labels(
    image_path: str,
    label_path: str,
    output_base_dir: str,
    global_target_size: Tuple[int, int] = (1280, 960),
    classification_target_size: Tuple[int, int] = (224, 224),
    fixed_pad_ratio: float = 0.125,
    fill_mode: str = 'mean',
    conf_threshold: Optional[float] = None
) -> Tuple[Dict[Any, int], int, List[str]]:
    """
    Processes a single image and its label file to generate and save padded crops.

    Args:
        image_path: Path to the input image.
        label_path: Path to the corresponding YOLO label file.
        output_base_dir: Base directory where class-specific folders for crops will be created.
        global_target_size: Tuple (width, height) for initial image resizing.
        classification_target_size: Tuple (width, height) for minimum crop size.
        fixed_pad_ratio: Ratio for fixed padding around detections.
        fill_mode: Method for background filling ('mean', 'median', 'gaussian').
        conf_threshold: Optional confidence threshold to filter detections.

    Returns:
        A tuple containing:
        - class_counts (Dict[Any, int]): Counts of crops generated per class_id.
        - total_crops (int): Total number of crops generated from this image.
        - saved_crop_paths (List[str]): List of paths to the saved crop images.
    """
    class_counts: Dict[Any, int] = {}
    total_crops = 0
    saved_crop_paths: List[str] = []

    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return class_counts, total_crops, saved_crop_paths

    detections = parse_label_file(label_path)
    if not detections:
        print(f"No valid detections found or label file error for {image_path}. Skipping.")
        return class_counts, total_crops, saved_crop_paths

    resized_image = _resize_image(image, global_target_size)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for idx, det in enumerate(detections):
        if conf_threshold is not None and 'conf' in det and det['conf'] is not None and det['conf'] < conf_threshold:
            continue

        norm_bbox = det['bbox']
        crop = crop_with_padding(
            resized_image,
            norm_bbox,
            fixed_pad_ratio,
            classification_target_size,
            fill_mode
        )

        class_id = det.get('class_id', "Unknown") # Use "Unknown" if class_id is missing
        class_folder = os.path.join(output_base_dir, f"Class_{class_id}")
        os.makedirs(class_folder, exist_ok=True)

        crop_filename = os.path.join(class_folder, f"{base_name}_crop{idx+1}.jpg")
        try:
            cv2.imwrite(crop_filename, crop)
            saved_crop_paths.append(crop_filename)
            total_crops += 1
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        except Exception as e:
            print(f"Error saving crop {crop_filename}: {e}")

    return class_counts, total_crops, saved_crop_paths

def batch_create_crops_from_dataset(
    image_folder: str,
    label_folder: str,
    output_base_dir: str,
    global_target_size: Tuple[int, int] = (1280, 960),
    classification_target_size: Tuple[int, int] = (224, 224),
    fixed_pad_ratio: float = 0.125,
    fill_mode: str = 'mean',
    conf_threshold: Optional[float] = None
) -> Tuple[Dict[Any, int], int]:
    """
    Processes all images in a folder to generate and save crops.

    Args:
        image_folder: Path to the folder containing images.
        label_folder: Path to the folder containing corresponding YOLO label files.
        output_base_dir: Base directory where class-specific folders for crops will be created.
        global_target_size: Tuple (width, height) for initial image resizing.
        classification_target_size: Tuple (width, height) for minimum crop size.
        fixed_pad_ratio: Ratio for fixed padding.
        fill_mode: Method for background filling.
        conf_threshold: Optional confidence threshold for detections.

    Returns:
        A tuple containing:
        - overall_class_counts (Dict[Any, int]): Aggregated counts of crops per class_id.
        - overall_total_crops (int): Total number of crops generated across all images.
    """
    img_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_files: List[str] = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))

    if not image_files:
        print(f"No images found in '{image_folder}'.")
        return {}, 0

    overall_class_counts: Dict[Any, int] = {}
    overall_total_crops = 0
    total_images_processed = 0

    for img_file_path in image_files:
        base_name = os.path.splitext(os.path.basename(img_file_path))[0]
        label_file_path = os.path.join(label_folder, base_name + ".txt")

        if not os.path.exists(label_file_path):
            print(f"Warning: Label file not found for {img_file_path}, skipping.")
            continue

        print(f"Processing {img_file_path}...")
        class_counts, num_crops, _ = create_crops_from_image_and_labels(
            img_file_path,
            label_file_path,
            output_base_dir,
            global_target_size,
            classification_target_size,
            fixed_pad_ratio,
            fill_mode,
            conf_threshold
        )

        if num_crops > 0:
            total_images_processed +=1
            overall_total_crops += num_crops
            for cls, count in class_counts.items():
                overall_class_counts[cls] = overall_class_counts.get(cls, 0) + count

    # Generate summary
    summary_lines = [
        f"Batch Cropping Summary:",
        f"------------------------",
        f"Total images processed (with detections): {total_images_processed}",
        f"Total crops saved: {overall_total_crops}",
        f"Crops per class:"
    ]
    for cls, count in sorted(overall_class_counts.items()): # Sorted for consistent output
        summary_lines.append(f"  Class {cls}: {count} crops")

    summary_text = "\n".join(summary_lines)
    summary_file_path = os.path.join(output_base_dir, "cropping_summary.txt")
    try:
        with open(summary_file_path, "w") as sf:
            sf.write(summary_text)
        print(f"\nSummary saved to {summary_file_path}")
    except Exception as e:
        print(f"Error writing summary file: {e}")

    print(f"\n{summary_text}")
    print("Batch cropping complete.")
    return overall_class_counts, overall_total_crops

# =============================================================================
# Main Block - Example Usage
# =============================================================================
if __name__ == '__main__':
    # --- Configuration ---
    # Set your input and output paths here
    IMAGE_FOLDER = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/val/images"
    LABEL_FOLDER = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/val/labels"
    OUTPUT_DIR = "/home/itk/Desktop/Andreas/AWAS-Project/Cropping/Refactored_Cropped_Images_Val"

    # Image processing parameters
    GLOBAL_TARGET_SIZE_CONFIG = (1280, 960)  # Initial resize (width, height)
    CLASSIFIER_TARGET_SIZE_CONFIG = (224, 224) # Min crop size (width, height) for padding reference

    # Padding and filling parameters
    FIXED_PAD_RATIO_CONFIG = 0.125  # 12.5% fixed padding
    FILL_MODE_CONFIG = 'median'     # 'mean', 'median', or 'gaussian'
    CONF_THRESHOLD_CONFIG = 0.0     # Minimum confidence for a detection to be processed (e.g., 0.25)
                                     # Set to None or 0.0 to include all detections.
    # --- End Configuration ---

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Starting batch image cropping process...")
    batch_create_crops_from_dataset(
        image_folder=IMAGE_FOLDER,
        label_folder=LABEL_FOLDER,
        output_base_dir=OUTPUT_DIR,
        global_target_size=GLOBAL_TARGET_SIZE_CONFIG,
        classification_target_size=CLASSIFIER_TARGET_SIZE_CONFIG,
        fixed_pad_ratio=FIXED_PAD_RATIO_CONFIG,
        fill_mode=FILL_MODE_CONFIG,
        conf_threshold=CONF_THRESHOLD_CONFIG
    )

    # --- Example of processing a single image (optional) ---
    # If you want to test with a single image:
    # Find an example image and label file from your dataset
    # example_image_name = "your_image_filename.jpg" # Replace with an actual image name
    # example_label_name = "your_image_filename.txt" # Replace with corresponding label name
    #
    # example_image_path = os.path.join(IMAGE_FOLDER, example_image_name)
    # example_label_path = os.path.join(LABEL_FOLDER, example_label_name)
    #
    # if os.path.exists(example_image_path) and os.path.exists(example_label_path):
    #     print(f"\nProcessing a single example image: {example_image_path}")
    #     counts, total, paths = create_crops_from_image_and_labels(
    #         image_path=example_image_path,
    #         label_path=example_label_path,
    #         output_base_dir=os.path.join(OUTPUT_DIR, "single_image_test"), # Save to a subfolder
    #         global_target_size=GLOBAL_TARGET_SIZE_CONFIG,
    #         classification_target_size=CLASSIFIER_TARGET_SIZE_CONFIG,
    #         fixed_pad_ratio=FIXED_PAD_RATIO_CONFIG,
    #         fill_mode=FILL_MODE_CONFIG,
    #         conf_threshold=CONF_THRESHOLD_CONFIG
    #     )
    #     print(f"Single image processing complete. Generated {total} crops.")
    #     print(f"Class counts: {counts}")
    #     # for p in paths:
    #     #     print(f"  Saved: {p}")
    # else:
    #     print(f"\nExample image/label for single test not found (check filenames).")