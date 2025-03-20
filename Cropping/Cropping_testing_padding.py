import cv2
import numpy as np
import os
import glob
import random

# =============================================================================
# Pipeline Overview (Notes to Self):
# - Input images are resized to a global target size (e.g., 1280×960).
# - YOLO annotations (center-based, normalized) are converted to normalized
#   corner-based coordinates.
# - For each detection:
#      1. Convert the normalized bbox to absolute coordinates.
#      2. Always add fixed padding: fixed_pad = S × fixed_pad_ratio (e.g., 12.5% of S)
#         to preserve extra context.
#      3. If the fixed-padded region is smaller than the desired minimum (224×224),
#         add dynamic padding (in pixels) to reach that size.
# - In production:
#      * Each crop is saved in a folder for its class (e.g., "Class_3") with filenames
#        like "OriginalName_cropN.jpg".
#      * A summary text file is generated with overall counts.
#      * A "random_samples" folder is generated with detailed evaluation for 1–4 images.
# =============================================================================

def compute_background(image, fill_mode='mean'):
    """
    Compute a background fill value from the full image.
    Options: 'mean', 'median', or 'gaussian'.
    Returns a BGR color tuple.
    """
    if fill_mode == 'mean':
        bg_color = np.mean(image, axis=(0, 1))
    elif fill_mode == 'median':
        bg_color = np.median(image, axis=(0, 1))
    elif fill_mode == 'gaussian':
        blurred = cv2.GaussianBlur(image, (31, 31), 0)
        center = blurred[blurred.shape[0] // 2, blurred.shape[1] // 2]
        bg_color = center
    else:
        bg_color = np.array([0, 0, 0])
    return tuple(map(int, bg_color))

def resize_image_and_bboxes(image, detections, target_size):
    """
    Resize the image to target_size (width, height).
    Normalized coordinates remain unchanged.
    """
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image, detections

def draw_bboxes_on_image(image, detections, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on a copy of the image.
    Detections are in normalized (corner) format and are converted on the fly.
    """
    im_copy = image.copy()
    h, w = im_copy.shape[:2]
    for det in detections:
        bbox = det.get('bbox')
        bbox_abs = (int(bbox[0]*w), int(bbox[1]*h), int(bbox[2]*w), int(bbox[3]*h))
        cv2.rectangle(im_copy, (bbox_abs[0], bbox_abs[1]), (bbox_abs[2], bbox_abs[3]), color, thickness)
        cv2.putText(im_copy, str(det.get('class_id', '')), (bbox_abs[0], bbox_abs[1]-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return im_copy

def crop_with_padding(image, norm_bbox, fixed_pad_ratio=0.125, classification_target_size=(224, 224), fill_mode='mean'):
    """
    Crop an instance from the image using a normalized (corner) bbox.
    
    Steps:
      1. Convert the normalized bbox to absolute coordinates.
      2. Compute fixed padding: fixed_pad = S × fixed_pad_ratio (S = max(width, height)).
         This fixed padding is always applied.
      3. If the fixed-padded region is smaller than the classifier target size,
         add dynamic padding (in pixels) based on the difference.
    
    Returns the final crop, which is guaranteed to be at least classification_target_size.
    (No further resizing is done.)
    """
    h, w = image.shape[:2]
    xmin = int(norm_bbox[0] * w)
    ymin = int(norm_bbox[1] * h)
    xmax = int(norm_bbox[2] * w)
    ymax = int(norm_bbox[3] * h)
    
    box_w = xmax - xmin
    box_h = ymax - ymin
    S = max(box_w, box_h)
    
    fixed_pad = int(S * fixed_pad_ratio)
    fixed_xmin = xmin - fixed_pad
    fixed_ymin = ymin - fixed_pad
    fixed_xmax = xmax + fixed_pad
    fixed_ymax = ymax + fixed_pad
    
    current_width = fixed_xmax - fixed_xmin
    current_height = fixed_ymax - fixed_ymin
    target_w, target_h = classification_target_size
    
    dynamic_pad_left = dynamic_pad_right = dynamic_pad_top = dynamic_pad_bottom = 0
    if current_width < target_w:
        extra_pad_x = target_w - current_width
        dynamic_pad_left = extra_pad_x // 2
        dynamic_pad_right = extra_pad_x - dynamic_pad_left
    if current_height < target_h:
        extra_pad_y = target_h - current_height
        dynamic_pad_top = extra_pad_y // 2
        dynamic_pad_bottom = extra_pad_y - dynamic_pad_top
    
    final_xmin = fixed_xmin - dynamic_pad_left
    final_ymin = fixed_ymin - dynamic_pad_top
    final_xmax = fixed_xmax + dynamic_pad_right
    final_ymax = fixed_ymax + dynamic_pad_bottom
    
    final_width = final_xmax - final_xmin
    final_height = final_ymax - final_ymin
    
    if final_xmin >= 0 and final_ymin >= 0 and final_xmax <= w and final_ymax <= h:
        crop = image[final_ymin:final_ymax, final_xmin:final_xmax].copy()
    else:
        bg_color = compute_background(image, fill_mode=fill_mode)
        crop = np.full((final_height, final_width, 3), bg_color, dtype=image.dtype)
        x1_src = max(final_xmin, 0)
        y1_src = max(final_ymin, 0)
        x2_src = min(final_xmax, w)
        y2_src = min(final_ymax, h)
        x1_dst = x1_src - final_xmin
        y1_dst = y1_src - final_ymin
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)
        crop[y1_dst:y2_dst, x1_dst:x2_dst] = image[y1_src:y2_src, x1_src:x2_src]
    return crop

def parse_label_file(label_path):
    """
    Parse a YOLO-format label file.
    
    Each line should contain 5 values:
         class_id, x_center, y_center, width, height
    or 6 values (the sixth is confidence).
    All coordinate values (except class_id) are normalized.
    
    Converts center-based annotations to normalized corner-based coordinates:
         x_min = x_center - (width/2)
         y_min = y_center - (height/2)
         x_max = x_center + (width/2)
         y_max = y_center + (height/2)
    
    Returns a list of detections with keys:
         'class_id', 'bbox' (normalized corner coordinates), and optionally 'conf'.
    """
    detections = []
    if not os.path.exists(label_path):
        print(f"Warning: Label file {label_path} not found.")
        return detections
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) not in [5, 6]:
                print("Skipping invalid annotation line:", line)
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width_norm = float(parts[3])
                height_norm = float(parts[4])
            except ValueError:
                print("Error parsing annotation line:", line)
                continue
            x_min = x_center - width_norm/2
            y_min = y_center - height_norm/2
            x_max = x_center + width_norm/2
            y_max = y_center + height_norm/2
            detection = {'class_id': class_id, 'bbox': (x_min, y_min, x_max, y_max)}
            if len(parts) == 6:
                try:
                    detection['conf'] = float(parts[5])
                except ValueError:
                    detection['conf'] = None
            detections.append(detection)
    return detections

# ---------------------------------------------------------------------------
# New: Compute final padded bbox (used in evaluation)
# ---------------------------------------------------------------------------
def compute_padded_bbox(norm_bbox, image_shape, fixed_pad_ratio, classification_target_size):
    """
    Compute the final padded bounding box (in absolute coordinates) from a normalized (corner) bbox.
    
    Steps:
      1. Convert normalized bbox to absolute coordinates.
      2. Compute fixed padding: fixed_pad = S × fixed_pad_ratio (S = max(width, height)).
         This fixed padding is always applied.
      3. If the fixed padded region is smaller than the classifier target size,
         add dynamic padding (in pixels) to meet the target.
    
    Returns (final_xmin, final_ymin, final_xmax, final_ymax).
    """
    h, w = image_shape[:2]
    xmin = int(norm_bbox[0]*w)
    ymin = int(norm_bbox[1]*h)
    xmax = int(norm_bbox[2]*w)
    ymax = int(norm_bbox[3]*h)
    box_w = xmax - xmin
    box_h = ymax - ymin
    S = max(box_w, box_h)
    
    fixed_pad = int(S * fixed_pad_ratio)
    fixed_xmin = xmin - fixed_pad
    fixed_ymin = ymin - fixed_pad
    fixed_xmax = xmax + fixed_pad
    fixed_ymax = ymax + fixed_pad
    
    current_width = fixed_xmax - fixed_xmin
    current_height = fixed_ymax - fixed_ymin
    target_w, target_h = classification_target_size
    dynamic_pad_left = dynamic_pad_right = dynamic_pad_top = dynamic_pad_bottom = 0
    if current_width < target_w:
        extra_pad_x = target_w - current_width
        dynamic_pad_left = extra_pad_x // 2
        dynamic_pad_right = extra_pad_x - dynamic_pad_left
    if current_height < target_h:
        extra_pad_y = target_h - current_height
        dynamic_pad_top = extra_pad_y // 2
        dynamic_pad_bottom = extra_pad_y - dynamic_pad_top
    
    final_xmin = fixed_xmin - dynamic_pad_left
    final_ymin = fixed_ymin - dynamic_pad_top
    final_xmax = fixed_xmax + dynamic_pad_right
    final_ymax = fixed_ymax + dynamic_pad_bottom
    return final_xmin, final_ymin, final_xmax, final_ymax

# ---------------------------------------------------------------------------
# Evaluation Mode (for testing and debug)
# ---------------------------------------------------------------------------
def evaluate_image(image_path, label_folder, output_eval_folder,
                   global_target_size, classification_target_size,
                   fixed_pad_ratio, fill_mode, conf_threshold):
    """
    Run evaluation on one image and document the process.
    
    Saves:
      - Original image with drawn normalized bboxes.
      - Resized image with drawn bboxes.
      - For each detection, an image showing:
            * Original absolute bbox (green),
            * Fixed padded bbox (blue),
            * Final padded bbox (red).
      - A metrics text file summarizing the process.
    """
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    label_file = os.path.join(label_folder, base_name + ".txt")
    detections = parse_label_file(label_file)
    if not detections:
        print("No valid detections for", image_path)
        return
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        print("Error loading", image_path)
        return
    
    os.makedirs(output_eval_folder, exist_ok=True)
    metrics_lines = []
    
    resized_img, _ = resize_image_and_bboxes(orig_img, detections, global_target_size)
    h_res, w_res = resized_img.shape[:2]
    
    summary = f"Global target size (input resize): {global_target_size}\n"
    summary += f"Resized image size: {w_res}x{h_res}\n"
    summary += f"Classifier target size (for padding reference): {classification_target_size}\n"
    summary += f"Fixed padding ratio: {fixed_pad_ratio*100:.1f}%\n\n"
    metrics_lines.append(summary)
    
    orig_with_bboxes = draw_bboxes_on_image(orig_img, detections, color=(0,255,0), thickness=2)
    cv2.imwrite(os.path.join(output_eval_folder, "original_with_bboxes.jpg"), orig_with_bboxes)
    resized_with_bboxes = draw_bboxes_on_image(resized_img, detections, color=(0,255,0), thickness=2)
    cv2.imwrite(os.path.join(output_eval_folder, "resized_with_bboxes.jpg"), resized_with_bboxes)
    
    for i, det in enumerate(detections):
        abs_bbox = (int(det['bbox'][0]*w_res), int(det['bbox'][1]*h_res),
                    int(det['bbox'][2]*w_res), int(det['bbox'][3]*h_res))
        box_w = abs_bbox[2] - abs_bbox[0]
        box_h = abs_bbox[3] - abs_bbox[1]
        S = max(box_w, box_h)
        fixed_pad = int(S * fixed_pad_ratio)
        fixed_padded_bbox = (abs_bbox[0] - fixed_pad, abs_bbox[1] - fixed_pad,
                             abs_bbox[2] + fixed_pad, abs_bbox[3] + fixed_pad)
        
        final_padded_bbox = compute_padded_bbox(det['bbox'], resized_img.shape, fixed_pad_ratio, classification_target_size)
        
        current_width = fixed_padded_bbox[2] - fixed_padded_bbox[0]
        current_height = fixed_padded_bbox[3] - fixed_padded_bbox[1]
        target_w, target_h = classification_target_size
        dynamic_pad_left = dynamic_pad_right = dynamic_pad_top = dynamic_pad_bottom = 0
        if current_width < target_w:
            extra_pad_x = target_w - current_width
            dynamic_pad_left = extra_pad_x // 2
            dynamic_pad_right = extra_pad_x - dynamic_pad_left
        if current_height < target_h:
            extra_pad_y = target_h - current_height
            dynamic_pad_top = extra_pad_y // 2
            dynamic_pad_bottom = extra_pad_y - dynamic_pad_top
        
        final_crop_width = final_padded_bbox[2] - final_padded_bbox[0]
        final_crop_height = final_padded_bbox[3] - final_padded_bbox[1]
        
        metrics_line = f"Detection {i+1}:\n"
        metrics_line += f"  Normalized bbox: {det['bbox']}\n"
        metrics_line += f"  Absolute bbox: {abs_bbox}\n"
        metrics_line += f"  Fixed padded bbox: {fixed_padded_bbox} (fixed_pad: {fixed_pad})\n"
        metrics_line += f"  Dynamic extra padding (pixels): left={dynamic_pad_left}, right={dynamic_pad_right}, top={dynamic_pad_top}, bottom={dynamic_pad_bottom}\n"
        metrics_line += f"  Final padded bbox: {final_padded_bbox}\n"
        metrics_line += f"  Final crop size (extracted from resized image): {final_crop_width}x{final_crop_height}\n\n"
        metrics_lines.append(metrics_line)
        
        eval_img = resized_img.copy()
        cv2.rectangle(eval_img, (abs_bbox[0], abs_bbox[1]), (abs_bbox[2], abs_bbox[3]), (0,255,0), 2)
        cv2.rectangle(eval_img, (fixed_padded_bbox[0], fixed_padded_bbox[1]), (fixed_padded_bbox[2], fixed_padded_bbox[3]), (255,0,0), 2)
        cv2.rectangle(eval_img, (final_padded_bbox[0], final_padded_bbox[1]), (final_padded_bbox[2], final_padded_bbox[3]), (0,0,255), 2)
        cv2.putText(eval_img, f"Det {i+1}", (abs_bbox[0], abs_bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)
        cv2.imwrite(os.path.join(output_eval_folder, f"detection_{i+1}_padded.jpg"), eval_img)
        
        crop = crop_with_padding(resized_img, det['bbox'], fixed_pad_ratio, classification_target_size, fill_mode)
        cv2.imwrite(os.path.join(output_eval_folder, f"detection_{i+1}_crop.jpg"), crop)
    
    metrics_file = os.path.join(output_eval_folder, "metrics.txt")
    with open(metrics_file, "w") as mf:
        mf.write("\n".join(metrics_lines))
    print(f"Evaluation images and metrics saved for {base_name} in {output_eval_folder}")

# ---------------------------------------------------------------------------
# Production Mode Functions
# ---------------------------------------------------------------------------
def crop_and_save_instances_production(image_path, detections, output_folder,
                            conf_threshold=None, global_target_size=(1280,960),
                            classification_target_size=(224,224),
                            fixed_pad_ratio=0.125, fill_mode='mean'):
    """
    Process a single image for production.
    
    1. Resize the image to the global target size.
    2. For each detection, compute the padded crop using:
         - Fixed padding: fixed_pad = S × fixed_pad_ratio (always applied).
         - Dynamic padding (in pixels) if the region is smaller than the target.
    3. Save each crop into a folder corresponding to its class ID (e.g., "Class_3") with filenames like "OriginalName_cropN.jpg".
    
    Returns a dictionary with species (class_id) counts and the total number of crops.
    """
    species_counts = {}
    total_crops = 0
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image {image_path}")
        return species_counts, total_crops
    resized_image, detections = resize_image_and_bboxes(image, detections, global_target_size)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    for idx, det in enumerate(detections):
        if conf_threshold is not None and det.get('conf') is not None and det.get('conf') < conf_threshold:
            continue
        crop = crop_with_padding(resized_image, det['bbox'], fixed_pad_ratio, classification_target_size, fill_mode)
        class_id = det.get('class_id', "Unknown")
        class_folder = os.path.join(output_folder, f"Class_{class_id}")
        os.makedirs(class_folder, exist_ok=True)
        crop_filename = os.path.join(class_folder, f"{base_name}_crop{idx+1}.jpg")
        cv2.imwrite(crop_filename, crop)
        total_crops += 1
        species_counts[class_id] = species_counts.get(class_id, 0) + 1
    return species_counts, total_crops

def process_images_with_labels_production(image_folder, label_folder, output_folder, conf_threshold=None,
                                          global_target_size=(1280,960), classification_target_size=(224,224),
                                          fixed_pad_ratio=0.125, fill_mode='mean'):
    """
    Process all images in image_folder for production.
    
    - Each crop is saved in a folder for its class.
    - A summary text file is generated with total images processed, total crops, and counts per species.
    """
    img_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    if not image_files:
        print("No images found in the provided image folder.")
        return
    overall_species_counts = {}
    overall_total_crops = 0
    total_images = 0
    for img_file in image_files:
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(label_folder, base_name + ".txt")
        detections = parse_label_file(label_file)
        if not detections:
            print(f"No valid detections for {img_file}. Skipping.")
            continue
        total_images += 1
        species_counts, crop_count = crop_and_save_instances_production(img_file, detections, output_folder,
                                                                        conf_threshold, global_target_size,
                                                                        classification_target_size, fixed_pad_ratio, fill_mode)
        overall_total_crops += crop_count
        for cls, count in species_counts.items():
            overall_species_counts[cls] = overall_species_counts.get(cls, 0) + count
    summary_lines = []
    summary_lines.append(f"Total images processed: {total_images}")
    summary_lines.append(f"Total crops saved: {overall_total_crops}")
    for cls, count in overall_species_counts.items():
        summary_lines.append(f"Class {cls}: {count} crops")
    summary_text = "\n".join(summary_lines)
    summary_file = os.path.join(output_folder, "production_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write(summary_text)
    print("Production processing complete.")
    print(summary_text)
    return overall_species_counts, overall_total_crops

# ---------------------------------------------------------------------------
# Mode Switch Functions
# ---------------------------------------------------------------------------
def run_evaluation(image_folder, label_folder, output_dir, num_images=4,
                   global_target_size=(1280,960), classification_target_size=(224,224),
                   fixed_pad_ratio=0.125, fill_mode='mean', conf_threshold=0.85):
    """
    Run evaluation mode on a random sample of num_images from image_folder.
    In production, if there are any images whose filenames start with a letter,
    3 out of the selected images will be chosen from that group (if available),
    and the remaining from other images.
    """
    img_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    if not image_files:
        print("No images found in", image_folder)
        return

    # Partition images into those starting with a letter and those starting with a digit.
    letter_images = [img for img in image_files if os.path.basename(img)[0].isalpha()]
    digit_images = [img for img in image_files if os.path.basename(img)[0].isdigit()]

    selected_images = []
    if letter_images:
        num_letter = min(3, len(letter_images))
        selected_images.extend(random.sample(letter_images, num_letter))
        remaining = num_images - num_letter
        # If there are digit images, choose from them; otherwise, fill with letter images.
        if remaining > 0:
            if digit_images:
                selected_images.extend(random.sample(digit_images, min(remaining, len(digit_images))))
            else:
                selected_images.extend(random.sample(letter_images, min(remaining, len(letter_images))))
    else:
        # If no letter images, choose randomly from all images.
        selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    for img_file in selected_images:
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        eval_folder = os.path.join(output_dir, base_name, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        evaluate_image(img_file, label_folder, eval_folder,
                       global_target_size, classification_target_size,
                       fixed_pad_ratio, fill_mode, conf_threshold)


def run_production(image_folder, label_folder, output_dir,
                   global_target_size=(1280,960), classification_target_size=(224,224),
                   fixed_pad_ratio=0.125, fill_mode='mean', conf_threshold=0.85):
    """
    Run production mode on all images in image_folder.
    
    Each crop is saved into a folder corresponding to its class.
    A summary text file is written with overall statistics.
    Additionally, a "random_samples" folder is created with a random subset of 1-4 images
    evaluated in full detail for quality control.
    """
    process_images_with_labels_production(image_folder, label_folder, output_dir, conf_threshold,
                                          global_target_size, classification_target_size,
                                          fixed_pad_ratio, fill_mode)
    sample_folder = os.path.join(output_dir, "random_samples")
    os.makedirs(sample_folder, exist_ok=True)
    run_evaluation(image_folder, label_folder, sample_folder, num_images=4,
                   global_target_size=global_target_size, classification_target_size=classification_target_size,
                   fixed_pad_ratio=fixed_pad_ratio, fill_mode=fill_mode, conf_threshold=conf_threshold)

# =============================================================================
# Main Block - Mode Switch
# Set mode = "evaluation" for sample testing, or "production" for full processing.
# =============================================================================
if __name__ == '__main__':
    # Updated paths:
    image_folder = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/images"
    label_folder = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/labels"
    output_dir = "/home/itk/Desktop/Andreas/AWAS-Project/Cropping/Cropped_images_multiple_classes_train"
    
    GLOBAL_TARGET_SIZE = (1280,960)
    CLASSIFIER_TARGET_SIZE = (224,224)  # For padding reference; final crop is not resized further.
    
    # Padding parameter: Fixed padding ratio (always applied) of 12.5%.
    FIXED_PAD_RATIO = 0.125
    FILL_MODE = 'median'
    
    CONF_THRESHOLD = 0.0  # Set threshold as needed.
    
    # Set mode: "evaluation" for sample testing, "production" for full processing.
    mode = "production"  # Change to "evaluation" for testing a few images.
    
    if mode == "evaluation":
        run_evaluation(image_folder, label_folder, output_dir, num_images=4,
                       global_target_size=GLOBAL_TARGET_SIZE, classification_target_size=CLASSIFIER_TARGET_SIZE,
                       fixed_pad_ratio=FIXED_PAD_RATIO, fill_mode=FILL_MODE, conf_threshold=CONF_THRESHOLD)
    elif mode == "production":
        run_production(image_folder, label_folder, output_dir,
                       global_target_size=GLOBAL_TARGET_SIZE, classification_target_size=CLASSIFIER_TARGET_SIZE,
                       fixed_pad_ratio=FIXED_PAD_RATIO, fill_mode=FILL_MODE, conf_threshold=CONF_THRESHOLD)
    else:
        print("Invalid mode. Choose 'evaluation' or 'production'.")


# =============================================================================
# Pipeline and Padding Strategy Summary:
#
# 1. Input images are resized to a global target size (e.g., 1280×960), and YOLO
#    annotations (center-based, normalized) are converted to normalized corner-based
#    coordinates.
#
# 2. For each detection, the normalized bbox is converted to absolute coordinates.
#
# 3. Fixed Padding:
#    - Always adds a fixed extra margin (12.5% of the object's max dimension) to all sides.
#    - This ensures extra context is always preserved around the object.
#
# 4. Dynamic Padding:
#    - If the fixed-padded region is still smaller than the minimum target (224×224),
#      additional dynamic padding (in pixels) is added to exactly meet the target size.
#
# 5. Background Filling:
#    - When padding extends beyond the image, missing areas are filled using the median 
#      of the image, which is robust to outliers in variable water conditions.
#
# =============================================================================


# =============================================================================
# Median Background Filling:
# - When padding extends beyond the image, missing areas need to be filled.
# - Using the median of the image for background filling is advantageous in underwater
#   settings (e.g., plankton images) where water conditions may vary and contain outliers
#   (like reflections or murkiness). The median is less sensitive to these outliers than the mean,
#   providing a more robust and representative background color.
# =============================================================================
