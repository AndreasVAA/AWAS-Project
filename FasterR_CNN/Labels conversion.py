import os
import cv2
import random
import matplotlib.pyplot as plt

def convert_yolo_to_minmax(x_center_norm, y_center_norm, w_norm, h_norm, img_width, img_height):
    """
    Convert normalized YOLO annotation to absolute min/max coordinates.
    
    Parameters:
      - x_center_norm, y_center_norm, w_norm, h_norm: Normalized values from YOLO.
      - img_width, img_height: Dimensions of the corresponding image.
      
    Returns:
      - A tuple (xmin, ymin, xmax, ymax) with absolute coordinates.
    """
    x_center = x_center_norm * img_width
    y_center = y_center_norm * img_height
    w = w_norm * img_width
    h = h_norm * img_height
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    xmax = x_center + w / 2
    ymax = y_center + h / 2
    return (xmin, ymin, xmax, ymax)

def convert_labels(images_dir, yolo_labels_dir, output_base_dir):
    """
    Converts YOLO-format labels to min/max format and saves them in a new folder.
    
    Parameters:
      - images_dir: Directory containing the image files.
      - yolo_labels_dir: Directory containing the original YOLO label files.
      - output_base_dir: Base directory where the new label folder will be created.
      
    The new coordinate format is "minmax" (absolute coordinates: xmin, ymin, xmax, ymax).
    The folder will be named "labels_minmax".
    
    *** IMPORTANT: The original YOLO labels remain unchanged. ***
    """
    # Define the output folder name based on the format type.
    output_folder_name = "labels_minmax"  # New coordinate format is "minmax"
    output_labels_dir = os.path.join(output_base_dir, output_folder_name)
    
    # Create the output folder if it doesn't exist.
    if not os.path.exists(output_labels_dir):
        os.makedirs(output_labels_dir)
    
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
    converted_count = 0
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        label_file = os.path.join(yolo_labels_dir, os.path.splitext(img_file)[0] + ".txt")
        
        # Skip if the corresponding YOLO label file doesn't exist.
        if not os.path.exists(label_file):
            continue

        # Read the image to get its dimensions.
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_file}. Skipping conversion for this file.")
            continue
        height, width = img.shape[:2]

        new_lines = []
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                # Do not change anything in the original YOLO labels folder.
                if len(parts) != 5:
                    continue  # Skip malformed lines.
                class_id = parts[0]
                x_center_norm, y_center_norm, w_norm, h_norm = map(float, parts[1:])
                xmin, ymin, xmax, ymax = convert_yolo_to_minmax(x_center_norm, y_center_norm, w_norm, h_norm, width, height)
                new_line = f"{class_id} {xmin:.2f} {ymin:.2f} {xmax:.2f} {ymax:.2f}\n"
                new_lines.append(new_line)
        
        # Write converted labels only if valid lines were found.
        if new_lines:
            output_file = os.path.join(output_labels_dir, os.path.splitext(img_file)[0] + ".txt")
            with open(output_file, 'w') as f_out:
                f_out.writelines(new_lines)
            converted_count += 1
            
    print(f"Converted labels for {converted_count} images and saved in '{output_labels_dir}'.")
    return output_labels_dir  # Return the new folder path for later use.

def count_labels_in_folder(labels_dir):
    """
    Counts the total number of valid label lines (5 tokens per line) in all text files within a folder.
    """
    total_labels = 0
    for file in os.listdir(labels_dir):
        if file.lower().endswith('.txt'):
            file_path = os.path.join(labels_dir, file)
            with open(file_path, 'r') as f:
                for line in f:
                    if len(line.strip().split()) == 5:
                        total_labels += 1
    return total_labels

def visualize_samples(images_dir, yolo_labels_dir, output_labels_dir, sample_size=6):
    """
    Visualizes a random sample of images with the converted bounding boxes.
    Also verifies that the number of bounding boxes in the converted label file
    matches the number in the original YOLO label file.
    
    Parameters:
      - images_dir: Directory containing image files.
      - yolo_labels_dir: Directory with the original YOLO label files.
      - output_labels_dir: Directory with converted label files.
      - sample_size: Number of images to sample for visualization.
    """
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png'))]
    sample_images = random.sample(image_files, min(sample_size, len(image_files)))
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for ax, img_file in zip(axes, sample_images):
        img_path = os.path.join(images_dir, img_file)
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Paths for original and converted labels.
        original_label_path = os.path.join(yolo_labels_dir, os.path.splitext(img_file)[0] + ".txt")
        converted_label_path = os.path.join(output_labels_dir, os.path.splitext(img_file)[0] + ".txt")
        
        # Count valid boxes in the original file.
        orig_count = 0
        if os.path.exists(original_label_path):
            with open(original_label_path, 'r') as f_orig:
                for line in f_orig:
                    if len(line.strip().split()) == 5:
                        orig_count += 1
        
        # Count and store boxes from the converted file.
        conv_count = 0
        boxes = []
        if os.path.exists(converted_label_path):
            with open(converted_label_path, 'r') as f_conv:
                for line in f_conv:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        conv_count += 1
                        boxes.append(parts)
        
        # Verify counts.
        if orig_count != conv_count:
            print(f"Warning: For image '{img_file}', original count {orig_count} != converted count {conv_count}")
        else:
            print(f"Verified '{img_file}': {orig_count} boxes converted correctly.")
        
        # Draw the converted bounding boxes.
        for box in boxes:
            # box format: class_id, xmin, ymin, xmax, ymax
            class_id, xmin, ymin, xmax, ymax = box
            cv2.rectangle(img, (int(float(xmin)), int(float(ymin))), (int(float(xmax)), int(float(ymax))), (0, 255, 0), 2)
            cv2.putText(img, class_id, (int(float(xmin)), int(float(ymin)) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Convert BGR image to RGB for display.
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_title(img_file)
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# --- Example Usage ---
if __name__ == '__main__':
    # Define your paths:
    #images_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images"              # Folder with image files.
    #yolo_labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/labels"    # Folder with original YOLO label files.
    #output_base_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train"  # Base folder where "labels_minmax" will be created.
    
    images_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/images"              # Folder with image files.
    yolo_labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels"    # Folder with original YOLO label files.
    output_base_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val"  # Base folder where "labels_minmax" will be created.
    # Convert labels (original YOLO files remain unchanged).
    converted_folder = convert_labels(images_dir, yolo_labels_dir, output_base_dir)
    
    # Report total counts.
    total_original = count_labels_in_folder(yolo_labels_dir)
    total_converted = count_labels_in_folder(converted_folder)
    print(f"\nTotal original labels: {total_original}")
    print(f"Total converted labels: {total_converted}\n")
    
    # Visualize sample images with converted bounding boxes.
    visualize_samples(images_dir, yolo_labels_dir, converted_folder, sample_size=6)
