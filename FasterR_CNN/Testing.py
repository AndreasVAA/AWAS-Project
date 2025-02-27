import cv2
import matplotlib.pyplot as plt

def draw_annotations(image_path, txt_path, output_path="annotated_image.jpg"):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Convert the image from BGR (OpenCV default) to RGB for correct display in matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Read the annotation file
    with open(txt_path, "r") as file:
        lines = file.readlines()

    # Process each line of the annotation file
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            print(f"Skipping invalid line: {line}")
            continue

        label = parts[0]
        try:
            xmin = int(float(parts[1]))
            ymin = int(float(parts[2]))
            xmax = int(float(parts[3]))
            ymax = int(float(parts[4]))
        except Exception as e:
            print(f"Error parsing line '{line}': {e}")
            continue

        # Draw the bounding box (green, thickness=2)
        cv2.rectangle(image_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        # Draw the label text above the bounding box
        cv2.putText(image_rgb, label, (xmin, max(ymin - 10, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the annotated image using matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.title("Annotated Image")
    plt.show()

    # Optionally, save the annotated image (convert back to BGR if needed)
    annotated_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, annotated_image)
    print(f"Annotated image saved to {output_path}")

if __name__ == "__main__":
    # Use your provided paths
    image_path = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images/1.jpg"
    txt_path = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/labels_minmax/1.txt"
    draw_annotations(image_path, txt_path)
