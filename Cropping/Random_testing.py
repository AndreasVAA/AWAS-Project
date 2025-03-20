import os
import glob
import cv2
import numpy as np
import random
import torch  # For GPU tensor operations
from ultralytics import SAM  # Ensure SAM is installed
from PIL import Image

# -------------------------
# Helper Functions (Common)
# -------------------------
def compute_background(image, fill_mode='mean'):
    """
    Compute a background fill color from the image.
    Options: 'mean', 'median', or 'gaussian'.
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
    Resize the image to the target size.
    Returns the resized image along with unchanged detections.
    """
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    return resized_image, detections

def parse_label_file(label_path):
    """
    Parse a YOLO-format label file into a list of detections.
    Each detection is a dict with keys 'class_id' and 'bbox' (normalized coordinates).
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
            x_min = x_center - width_norm / 2
            y_min = y_center - height_norm / 2
            x_max = x_center + width_norm / 2
            y_max = y_center + height_norm / 2
            detection = {'class_id': class_id, 'bbox': (x_min, y_min, x_max, y_max)}
            if len(parts) == 6:
                try:
                    detection['conf'] = float(parts[5])
                except ValueError:
                    detection['conf'] = None
            detections.append(detection)
    return detections

def draw_bboxes_on_image(image, detections, color=(0, 255, 0), thickness=2):
    """
    Draw bounding boxes on a copy of the image.
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

# -------------------------
# New Function: Mask ROI with Background
# -------------------------
def mask_roi_with_background(roi, mask, fill_mode='median'):
    """
    Given:
      - roi: region of interest (numpy array of shape (H, W, 3))
      - mask: binary mask (shape (H, W)) with 255 for object and 0 for background
    Replace background pixels in the ROI with a computed background color.
    """
    bg_color = compute_background(roi, fill_mode)
    object_mask = mask > 0
    masked_roi = roi.copy()
    masked_roi[~object_mask] = bg_color
    return masked_roi

# -------------------------
# Functions for Prompts (No Padding)
# -------------------------
def convert_bbox_no_padding(norm_bbox, image_shape):
    """
    Convert normalized bbox to absolute coordinates (no padding).
    Returns (xmin, ymin, xmax, ymax).
    """
    h, w = image_shape[:2]
    xmin = int(norm_bbox[0]*w)
    ymin = int(norm_bbox[1]*h)
    xmax = int(norm_bbox[2]*w)
    ymax = int(norm_bbox[3]*h)
    return xmin, ymin, xmax, ymax

def get_prompts(image_shape, bbox):
    """
    For a given absolute bbox (xmin, ymin, xmax, ymax) and image shape,
    return:
      - A list of prompt points: one positive point (center of the bbox) and negative points (corners outside bbox).
      - A list of corresponding labels (1 for positive, 0 for negative).
    """
    h, w = image_shape[:2]
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    positive_point = [cx, cy]
    neg_points = []
    corners = [(0, 0), (w-1, 0), (0, h-1), (w-1, h-1)]
    for x, y in corners:
        if x < bbox[0] or x > bbox[2] or y < bbox[1] or y > bbox[3]:
            neg_points.append([x, y])
    points = [positive_point] + neg_points
    labels = [1] + [0] * len(neg_points)
    return points, labels

# -------------------------
# Updated SAM Segmentation Function (No Padding, with Negative Points)
# -------------------------
def segment_instance_with_sam(image, norm_bbox, sam_model, fixed_pad_ratio, classification_target_size, fill_mode='mean'):
    """
    For a given normalized bbox, convert it to absolute coordinates (no padding) and run SAM segmentation.
    Uses the bounding box as a positive prompt along with additional prompt points.
    The prompt points and labels are unsqueezed to add a batch dimension.
    """
    device = next(sam_model.parameters()).device
    bbox_abs = convert_bbox_no_padding(norm_bbox, image.shape)
    bbox_tensor = torch.tensor(list(bbox_abs), dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 4)
    points, labels = get_prompts(image.shape, bbox_abs)  # e.g., 5 points total
    points_tensor = torch.tensor(points, dtype=torch.float32, device=device).unsqueeze(0)  # Shape: (1, 5, 2)
    labels_tensor = torch.tensor(labels, dtype=torch.int64, device=device).unsqueeze(0)    # Shape: (1, 5)
    results = sam_model(image, bboxes=bbox_tensor, points=points_tensor, labels=labels_tensor)
    return results[0] if isinstance(results, list) else results

# -------------------------
# Production Pipeline Functions
# -------------------------
def process_single_image_production_SAM(image_path, detections, sam_model, output_folder,
                                        conf_threshold=None, global_target_size=(1280,960),
                                        classification_target_size=(224,224),
                                        fixed_pad_ratio=0.125, fill_mode='median'):
    """
    Process a single image for production:
      - Resize the image.
      - For each detection, run SAM segmentation using the bounding box (no padding) and negative prompts.
      - Save:
          • Segmentation mask image.
          • ROI masked image: the original bounding box region with background fill outside the mask.
    Returns species counts and number of processed detections.
    """
    species_counts = {}
    total_segmentations = 0
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image {image_path}")
        return species_counts, total_segmentations

    resized_image, detections = resize_image_and_bboxes(image, detections, global_target_size)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for idx, det in enumerate(detections):
        if conf_threshold is not None and det.get('conf') is not None and det.get('conf') < conf_threshold:
            continue

        seg_result = segment_instance_with_sam(resized_image, det['bbox'], sam_model,
                                                 fixed_pad_ratio, classification_target_size, fill_mode)
        class_id = det.get('class_id', "Unknown")
        class_folder = os.path.join(output_folder, f"Class_{class_id}")
        os.makedirs(class_folder, exist_ok=True)

        # Save segmentation mask image.
        segmask_filename = os.path.join(class_folder, f"{base_name}_d{idx+1}_segmask.png")
        try:
            if seg_result.masks is not None:
                mask = seg_result.masks.data[0]
                mask = (mask > 0.5).to(torch.uint8) * 255
                mask_cpu = mask.cpu().numpy()
                cv2.imwrite(segmask_filename, mask_cpu)
                print(f"Saved segmentation mask for detection {idx+1} to {segmask_filename}")
            else:
                print(f"No mask available for detection {idx+1}")
        except Exception as e:
            print(f"Error saving segmentation mask for detection {idx+1}: {e}")
            continue

        # Extract ROI using the original bounding box.
        bbox_abs = convert_bbox_no_padding(det['bbox'], resized_image.shape)
        x_min, y_min, x_max, y_max = bbox_abs
        roi = resized_image[y_min:y_max, x_min:x_max].copy()
        if seg_result.masks is not None:
            mask_cpu = mask.cpu().numpy()
            mask_roi = mask_cpu[y_min:y_max, x_min:x_max]
            roi_masked = mask_roi_with_background(roi, mask_roi, fill_mode)
            roi_masked_filename = os.path.join(class_folder, f"{base_name}_d{idx+1}_roi_masked.jpg")
            cv2.imwrite(roi_masked_filename, roi_masked)
            print(f"Saved ROI masked image for detection {idx+1} to {roi_masked_filename}")
        else:
            print(f"Could not create ROI masked image for detection {idx+1} due to missing mask.")

        total_segmentations += 1
        species_counts[class_id] = species_counts.get(class_id, 0) + 1

    return species_counts, total_segmentations

def run_production_pipeline_SAM(image_folder, label_folder, output_folder, sam_model,
                                conf_threshold=None, global_target_size=(1280,960),
                                classification_target_size=(224,224),
                                fixed_pad_ratio=0.125, fill_mode='median'):
    """
    Run the production pipeline:
      - Process all images in the folder.
      - For each image, parse detections and run segmentation.
      - Save segmentation mask images and ROI masked images in class-based folders.
      - Produce a summary text file with overall counts.
    """
    img_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    if not image_files:
        print("No images found in the provided image folder.")
        return

    overall_species_counts = {}
    overall_total_seg = 0
    total_images = 0

    for img_file in image_files:
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        label_file = os.path.join(label_folder, base_name + ".txt")
        detections = parse_label_file(label_file)
        if not detections:
            print(f"No valid detections for {img_file}. Skipping.")
            continue

        total_images += 1
        species_counts, seg_count = process_single_image_production_SAM(
            img_file, detections, sam_model, output_folder,
            conf_threshold, global_target_size, classification_target_size,
            fixed_pad_ratio, fill_mode
        )
        overall_total_seg += seg_count
        for cls, count in species_counts.items():
            overall_species_counts[cls] = overall_species_counts.get(cls, 0) + count

    summary_lines = [
        f"Total images processed: {total_images}",
        f"Total segmentations saved: {overall_total_seg}"
    ]
    for cls, count in overall_species_counts.items():
        summary_lines.append(f"Class {cls}: {count} segmentations")
    summary_text = "\n".join(summary_lines)
    summary_file = os.path.join(output_folder, "production_summary.txt")
    with open(summary_file, "w") as sf:
        sf.write(summary_text)
    print("Production pipeline processing complete.")
    print(summary_text)
    return overall_species_counts, overall_total_seg

# -------------------------
# Evaluation Pipeline Functions
# -------------------------
def process_single_image_evaluation_SAM(image_path, label_folder, output_eval_folder, sam_model,
                                        global_target_size=(1280,960), classification_target_size=(224,224),
                                        fixed_pad_ratio=0.125, fill_mode='median', conf_threshold=0.85):
    """
    Process a single image for evaluation:
      - Resize image and draw YOLO boxes.
      - For each detection, run segmentation and save:
          • Segmentation mask image.
          • ROI masked image (using original bounding box with background fill).
          • An evaluation image with the bounding box drawn.
      - Also create a composite image of all segmentation masks overlaid on the resized image.
      - Save a metrics text file with details.
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

    summary = f"Global target size: {global_target_size}\n"
    summary += f"Resized image size: {w_res}x{h_res}\n"
    summary += f"Classifier target size: {classification_target_size}\n"
    summary += "Using no padding for segmentation.\n\n"
    metrics_lines.append(summary)

    orig_boxes_filename = os.path.join(output_eval_folder, f"{base_name}_original_with_boxes.jpg")
    resized_boxes_filename = os.path.join(output_eval_folder, f"{base_name}_resized_with_boxes.jpg")
    orig_with_boxes = draw_bboxes_on_image(orig_img, detections, color=(0,255,0), thickness=2)
    cv2.imwrite(orig_boxes_filename, orig_with_boxes)
    resized_with_boxes = draw_bboxes_on_image(resized_img, detections, color=(0,255,0), thickness=2)
    cv2.imwrite(resized_boxes_filename, resized_with_boxes)

    composite = resized_img.copy()

    for i, det in enumerate(detections):
        if conf_threshold is not None and det.get('conf') is not None and det.get('conf') < conf_threshold:
            continue

        bbox_abs = convert_bbox_no_padding(det['bbox'], resized_img.shape)
        seg_result = segment_instance_with_sam(resized_img, det['bbox'], sam_model,
                                                 fixed_pad_ratio, classification_target_size, fill_mode)
        eval_img = resized_img.copy()
        cv2.rectangle(eval_img, (bbox_abs[0], bbox_abs[1]), (bbox_abs[2], bbox_abs[3]), (0,255,0), 2)
        cv2.putText(eval_img, f"Det {i+1}", (bbox_abs[0], bbox_abs[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        segmask_filename = os.path.join(output_eval_folder, f"{base_name}_d{i+1}_segmask.png")
        if seg_result.masks is not None:
            mask = seg_result.masks.data[0]
            mask = (mask > 0.5).to(torch.uint8) * 255
            mask_cpu = mask.cpu().numpy()
            cv2.imwrite(segmask_filename, mask_cpu)
            colored_mask = cv2.applyColorMap(mask_cpu, cv2.COLORMAP_JET)
            composite = cv2.addWeighted(composite, 1.0, colored_mask, 0.5, 0)
        else:
            segmask_filename = "No mask available"

        eval_filename = os.path.join(output_eval_folder, f"{base_name}_d{i+1}_eval.jpg")
        cv2.imwrite(eval_filename, eval_img)

        roi = resized_img[bbox_abs[1]:bbox_abs[3], bbox_abs[0]:bbox_abs[2]].copy()
        if seg_result.masks is not None:
            mask_cpu = mask.cpu().numpy()
            mask_roi = mask_cpu[bbox_abs[1]:bbox_abs[3], bbox_abs[0]:bbox_abs[2]]
            roi_masked = mask_roi_with_background(roi, mask_roi, fill_mode)
            roi_masked_filename = os.path.join(output_eval_folder, f"{base_name}_d{i+1}_roi_masked.jpg")
            cv2.imwrite(roi_masked_filename, roi_masked)
        else:
            roi_masked_filename = "No masked ROI"

        metrics_line = f"Detection {i+1}:\n"
        metrics_line += f"  Normalized bbox: {det['bbox']}\n"
        metrics_line += f"  Absolute bbox: {bbox_abs}\n"
        metrics_line += f"  Segmentation mask saved as: {segmask_filename}\n"
        metrics_line += f"  ROI masked image saved as: {roi_masked_filename}\n\n"
        metrics_lines.append(metrics_line)

    composite_filename = os.path.join(output_eval_folder, f"composite_segmentation_{base_name}.jpg")
    cv2.imwrite(composite_filename, composite)
    metrics_lines.append(f"Composite segmentation overlay saved as: {composite_filename}\n")

    metrics_file = os.path.join(output_eval_folder, f"{base_name}_SAM_metrics.txt")
    with open(metrics_file, "w") as mf:
        mf.write("\n".join(metrics_lines))
    print(f"SAM evaluation images and metrics saved for {base_name} in {output_eval_folder}")

def run_evaluation_pipeline_SAM(image_folder, label_folder, output_dir, sam_model, num_images=4,
                                global_target_size=(1280,960), classification_target_size=(224,224),
                                fixed_pad_ratio=0.125, fill_mode='median', conf_threshold=0.85):
    """
    Run the evaluation pipeline on a subset of images:
      - Randomly select a set (e.g., 4 images).
      - For each image, process and save evaluation outputs.
    """
    img_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    if not image_files:
        print("No images found in", image_folder)
        return

    # Partition images: select up to 3 starting with a letter, then the rest from digits.
    letter_images = [img for img in image_files if os.path.basename(img)[0].isalpha()]
    digit_images = [img for img in image_files if os.path.basename(img)[0].isdigit()]

    selected_images = []
    if letter_images:
        num_letter = min(3, len(letter_images))
        selected_images.extend(random.sample(letter_images, num_letter))
        remaining = num_images - num_letter
        if remaining > 0:
            if digit_images:
                selected_images.extend(random.sample(digit_images, min(remaining, len(digit_images))))
            else:
                selected_images.extend(random.sample(letter_images, min(remaining, len(letter_images))))
    else:
        selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    for img_file in selected_images:
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        eval_folder = os.path.join(output_dir, base_name, "SAM_evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        process_single_image_evaluation_SAM(img_file, label_folder, eval_folder, sam_model,
                                            global_target_size, classification_target_size,
                                            fixed_pad_ratio, fill_mode, conf_threshold)

# -------------------------
# Main Block - Mode Switch
# -------------------------
if __name__ == '__main__':
    # Updated paths:
    image_folder = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/images"
    label_folder = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/labels"
    output_dir = "Entirely_NEW_SAM_output_testnig_evaluation"

    GLOBAL_TARGET_SIZE = (1280, 960)
    CLASSIFIER_TARGET_SIZE = (224, 224)
    FIXED_PAD_RATIO = 0.0  # Not used now.
    FILL_MODE = 'median'   # Used for background fill.
    CONF_THRESHOLD = 0.0   # Adjust threshold if needed

    # Load SAM model once.
    sam_model_path = "sam_b.pt"  # Update with your SAM model weight path/name.
    sam_model = SAM(sam_model_path)
    if torch.cuda.is_available():
        sam_model = sam_model.cuda()
        print("Using GPU for inference.")
    else:
        print("GPU not available. Using CPU.")

    sam_model.info()

    # Mode: choose "evaluation" or "production".
    mode = "evaluation"  # Change to "production" as needed.

    if mode == "evaluation":
        run_evaluation_pipeline_SAM(image_folder, label_folder, output_dir, sam_model,
                                    num_images=4, global_target_size=GLOBAL_TARGET_SIZE,
                                    classification_target_size=CLASSIFIER_TARGET_SIZE,
                                    fixed_pad_ratio=FIXED_PAD_RATIO, fill_mode=FILL_MODE,
                                    conf_threshold=CONF_THRESHOLD)
    elif mode == "production":
        run_production_pipeline_SAM(image_folder, label_folder, output_dir, sam_model,
                                    conf_threshold=CONF_THRESHOLD, global_target_size=GLOBAL_TARGET_SIZE,
                                    classification_target_size=CLASSIFIER_TARGET_SIZE,
                                    fixed_pad_ratio=FIXED_PAD_RATIO, fill_mode=FILL_MODE)
        random_samples_folder = os.path.join(output_dir, "random_samples")
        os.makedirs(random_samples_folder, exist_ok=True)
        run_evaluation_pipeline_SAM(image_folder, label_folder, random_samples_folder, sam_model,
                                    num_images=4, global_target_size=GLOBAL_TARGET_SIZE,
                                    classification_target_size=CLASSIFIER_TARGET_SIZE,
                                    fixed_pad_ratio=FIXED_PAD_RATIO, fill_mode=FILL_MODE,
                                    conf_threshold=CONF_THRESHOLD)
    else:
        print("Invalid mode. Choose 'evaluation' or 'production'.")






## I am currenlty passing bounding box, postive ponit in center and negative point in corners ringht? Doubel chekc rath thi9s ended up as stategy. 

## I might have to consdoer adding more htan just hte bounding box with backgound for getting clooser to image size of classifier .