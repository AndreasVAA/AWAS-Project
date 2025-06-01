import os

def count_instances_in_labels_folder(labels_dir):
    """
    Counts the total number of object instances (lines) in all .txt files
    within the specified labels directory.

    An instance is defined as a line with 5 parts (id + 4 coordinates).
    """
    total_instances = 0
    num_label_files = 0

    if not os.path.isdir(labels_dir):
        print(f"Error: Directory not found - {labels_dir}")
        return None, None

    for filename in os.listdir(labels_dir):
        if filename.lower().endswith(".txt"):
            num_label_files += 1
            file_path = os.path.join(labels_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) == 5: # Expecting class_id + 4 coordinates
                            total_instances += 1
                        elif line.strip(): # If line is not empty but not 5 parts
                            print(f"Warning: Malformed line in {filename}: '{line.strip()}'")
            except Exception as e:
                print(f"Error reading file {filename}: {e}")
    
    return total_instances, num_label_files

# Specify the directory containing your label files
labels_directory = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels_minmax"

# Count the instances
total_annotations, found_label_files = count_instances_in_labels_folder(labels_directory)

if total_annotations is not None:
    print(f"\nScanned {found_label_files} .txt files in the directory.")
    print(f"Total number of instances (bounding box annotations): {total_annotations}")