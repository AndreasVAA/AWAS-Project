import cv2
import numpy as np
import os

def compute_background(image, fill_mode='mean'):
    """
    Compute a background fill value from the full image.
    fill_mode: 'mean', 'median', or 'gaussian'
    Returns a BGR color tuple.
    """
    if fill_mode == 'mean':
        bg_color = np.mean(image, axis=(0, 1))
    elif fill_mode == 'median':
        bg_color = np.median(image, axis=(0, 1))
    elif fill_mode == 'gaussian':
        # Use a blurred version to extract a representative color from the center.
        blurred = cv2.GaussianBlur(image, (31, 31), 0)
        center = blurred[blurred.shape[0] // 2, blurred.shape[1] // 2]
        bg_color = center
    else:
        # Default to black if not specified
        bg_color = np.array([0, 0, 0])
    return tuple(map(int, bg_color))


def convert_bbox_normalized(bbox, image_shape):
    """
    Convert normalized bbox (xmin, ymin, xmax, ymax with values in [0,1])
    to absolute coordinates based on the image_shape.
    """
    h, w = image_shape[:2]
    return (int(bbox[0] * w), int(bbox[1] * h), int(bbox[2] * w), int(bbox[3] * h))


def resize_image_and_bboxes(image, bboxes, target_size, normalized=False):
    """
    Resize the full image to the target_size (width, height) and adjust bounding boxes.
    
    Parameters:
      - image: original image.
      - bboxes: list of detection dicts with key 'bbox' (and optionally 'conf').
      - target_size: (width, height) for global resizing.
      - normalized: if True, assume bboxes are normalized; else, they are in absolute coordinates.
    
    Returns:
      - resized_image: the resized image.
      - new_bboxes: list of detections with updated bounding boxes.
    """
    old_h, old_w = image.shape[:2]
    target_w, target_h = target_size
    resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)
    
    new_bboxes = []
    for det in bboxes:
        bbox = det.get('bbox')
        conf = det.get('conf', None)
        if normalized:
            # Use the target image dimensions to convert normalized values.
            new_bbox = (int(bbox[0]*target_w), int(bbox[1]*target_h), 
                        int(bbox[2]*target_w), int(bbox[3]*target_h))
        else:
            # Scale absolute coordinates based on the resize factor.
            scale_x = target_w / old_w
            scale_y = target_h / old_h
            new_bbox = (int(bbox[0]*scale_x), int(bbox[1]*scale_y), 
                        int(bbox[2]*scale_x), int(bbox[3]*scale_y))
        new_det = {'bbox': new_bbox}
        if conf is not None:
            new_det['conf'] = conf
        new_bboxes.append(new_det)
    return resized_image, new_bboxes


def crop_with_padding_dynamic(image, bbox, padding_ratio=0.125, classification_target_size=(224, 224), fill_mode='mean'):
    """
    Crop an instance from the image using the bounding box with dynamic padding.
    
    The padding is computed in two parts:
      1. A base padding defined as a fraction (padding_ratio) of the largest dimension of the bbox.
      2. Additional padding is added if the resulting crop is smaller than the target classification size.
    
    If the padded region extends beyond the image boundaries, the missing parts are filled using 
    the specified fill_mode (mean, median, gaussian) from the image.
    
    Parameters:
      - image: the resized image.
      - bbox: tuple (xmin, ymin, xmax, ymax) in absolute coordinates.
      - padding_ratio: base fraction of the bbox size to pad.
      - classification_target_size: (width, height) that the classifier expects.
      - fill_mode: method for background fill if the crop extends outside the image.
    
    Returns:
      - The cropped (and padded) instance as an image.
    """
    xmin, ymin, xmax, ymax = map(int, bbox)
    box_w = xmax - xmin
    box_h = ymax - ymin
    S = max(box_w, box_h)
    base_pad = int(S * padding_ratio)
    
    # Initial padded bbox
    new_xmin = xmin - base_pad
    new_ymin = ymin - base_pad
    new_xmax = xmax + base_pad
    new_ymax = ymax + base_pad
    
    # Calculate current padded dimensions
    current_width = new_xmax - new_xmin
    current_height = new_ymax - new_ymin
    target_w, target_h = classification_target_size
    
    # If the current crop is smaller than target, compute extra padding needed.
    extra_pad_x = max(0, target_w - current_width)
    extra_pad_y = max(0, target_h - current_height)
    
    # Distribute extra padding evenly
    extra_pad_left = extra_pad_x // 2
    extra_pad_right = extra_pad_x - extra_pad_left
    extra_pad_top = extra_pad_y // 2
    extra_pad_bottom = extra_pad_y - extra_pad_top
    
    # Update padded bbox coordinates
    new_xmin -= extra_pad_left
    new_xmax += extra_pad_right
    new_ymin -= extra_pad_top
    new_ymax += extra_pad_bottom

    final_width = new_xmax - new_xmin
    final_height = new_ymax - new_ymin

    img_h, img_w = image.shape[:2]
    # If the padded region is completely inside the image, crop directly.
    if new_xmin >= 0 and new_ymin >= 0 and new_xmax <= img_w and new_ymax <= img_h:
        crop = image[new_ymin:new_ymax, new_xmin:new_xmax].copy()
    else:
        # Create a blank canvas of the desired size, fill with background.
        bg_color = compute_background(image, fill_mode=fill_mode) if fill_mode is not None else (0, 0, 0)
        crop = np.full((final_height, final_width, 3), bg_color, dtype=image.dtype)
        # Determine overlapping region with the original image.
        x1_src = max(new_xmin, 0)
        y1_src = max(new_ymin, 0)
        x2_src = min(new_xmax, img_w)
        y2_src = min(new_ymax, img_h)
        # Corresponding destination coordinates in the crop.
        x1_dst = x1_src - new_xmin
        y1_dst = y1_src - new_ymin
        x2_dst = x1_dst + (x2_src - x1_src)
        y2_dst = y1_dst + (y2_src - y1_src)
        crop[y1_dst:y2_dst, x1_dst:x2_dst] = image[y1_src:y2_src, x1_src:x2_src]
    return crop


def crop_and_save_instances(image_path, bboxes, output_folder,
                            conf_threshold=None, global_target_size=(1280, 960),
                            classification_target_size=(224, 224),
                            normalized=False, padding_ratio=0.125, fill_mode='mean'):
    """
    Full pipeline to:
      1. Resize the full image (and adjust bounding boxes) to global_target_size.
      2. For each detection, apply dynamic cropping with padding.
      3. Finally, resize each crop to classification_target_size and save.
    
    Parameters:
      - image_path: Path to the input image.
      - bboxes: List of detections. Each detection is a dict with:
          'bbox': (xmin, ymin, xmax, ymax) [normalized if normalized=True, else absolute].
          'conf': (optional) confidence score.
      - output_folder: Directory to save cropped images.
      - conf_threshold: Minimum confidence to process a detection.
      - global_target_size: (width, height) to which the full image is resized (e.g. 1280x960).
      - classification_target_size: (width, height) for classifier input (e.g. 224x224).
      - normalized: Set True if the input bounding boxes are normalized.
      - padding_ratio: Base fraction for padding (e.g. 0.125 means S/8 padding).
      - fill_mode: 'mean', 'median', or 'gaussian' to fill missing regions.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Read original image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: could not load image {image_path}")
        return
    
    # Global resize of the image and adjust bounding boxes
    resized_image, new_bboxes = resize_image_and_bboxes(image, bboxes, global_target_size, normalized)
    
    crop_counter = 0
    for det in new_bboxes:
        bbox = det.get('bbox')
        conf = det.get('conf', None)
        
        if conf_threshold is not None and conf is not None and conf < conf_threshold:
            continue  # Skip low-confidence detections
        
        # Crop with dynamic padding: the crop is computed on the globally resized image.
        crop = crop_with_padding_dynamic(resized_image, bbox, padding_ratio, classification_target_size, fill_mode)
        # Finally, ensure the crop is exactly the target size for classification.
        crop_resized = cv2.resize(crop, classification_target_size, interpolation=cv2.INTER_LINEAR)
        
        crop_filename = os.path.join(output_folder, f"crop_{crop_counter:04d}.jpg")
        cv2.imwrite(crop_filename, crop_resized)
        crop_counter += 1
        
    print(f"Saved {crop_counter} cropped instances to {output_folder}")


# === Example Usage ===
if __name__ == '__main__':
    # Set your paths and directories.
    img_path = "path/to/your/image.jpg"
    output_dir = "path/to/output/folder"

    # Example detections:
    example_bboxes = [
        {"bbox": (0.04, 0.1, 0.15, 0.35), "conf": 0.95},  # Normalized coordinates example
        {"bbox": (0.17, 0.12, 0.29, 0.33), "conf": 0.80},
        # If using absolute coordinates (Pascal VOC), set normalized=False and provide pixel values.
        # {"bbox": (50, 100, 200, 300)},
    ]
    
    # Global image resolution (e.g., for detection) and classifier input resolution.
    GLOBAL_TARGET_SIZE = (1280, 960)    # (width, height)
    CLASSIFIER_TARGET_SIZE = (224, 224)  # (width, height)

    # Set normalized to True if using normalized bbox values; otherwise, False.
    use_normalized = True
    
    # Call the function. You can set conf_threshold to filter out low-confidence detections.
    crop_and_save_instances(img_path, example_bboxes, output_dir,
                            conf_threshold=0.85,
                            global_target_size=GLOBAL_TARGET_SIZE,
                            classification_target_size=CLASSIFIER_TARGET_SIZE,
                            normalized=use_normalized,
                            padding_ratio=0.125,
                            fill_mode='mean')
