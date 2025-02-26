import os

# Path to your train labels directory
labels_dir = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels_minmax"

# Initialize global counters and class set for final check
total_annotations_before = 0
total_annotations_after = 0
all_classes = set()

# Iterate over all files in the directory
for filename in os.listdir(labels_dir):
    if filename.lower().endswith(".txt"):
        file_path = os.path.join(labels_dir, filename)
        
        with open(file_path, "r") as file:
            lines = file.readlines()

        new_lines = []
        modified = False
        
        # Count annotations (only counting lines that are non-empty and properly formatted)
        file_count_before = 0
        file_count_after = 0
        
        for line in lines:
            stripped_line = line.strip()
            if not stripped_line:
                new_lines.append(line)
                continue

            parts = stripped_line.split()
            # Only process lines with exactly 5 tokens
            if len(parts) != 5:
                new_lines.append(line)
                continue

            file_count_before += 1

            # Check and update class id if needed
            if parts[0] == "0":
                parts[0] = "1"
                modified = True

            # Keep track of classes for the final check
            all_classes.add(parts[0])
            
            # Reassemble the line and add a newline character
            new_line = " ".join(parts) + "\n"
            new_lines.append(new_line)
            file_count_after += 1

        # Verify that annotation counts match before and after processing
        if file_count_before != file_count_after:
            print(f"Error in file {filename}: annotation count before ({file_count_before}) != after ({file_count_after})")
        else:
            print(f"File {filename}: {file_count_before} annotations processed, unchanged count.")

        total_annotations_before += file_count_before
        total_annotations_after += file_count_after

        # Write the modified content back if any changes were made
        if modified:
            with open(file_path, "w") as file:
                file.writelines(new_lines)
            print(f"Updated file: {filename}")
        else:
            print(f"No update needed for: {filename}")

print("\n=== Summary ===")
print(f"Total annotations before processing: {total_annotations_before}")
print(f"Total annotations after processing:  {total_annotations_after}")

if total_annotations_before != total_annotations_after:
    print("Error: The total number of annotations changed during processing!")
else:
    print("Success: Total number of annotations remained the same across all files.")

print(f"Unique class IDs found: {all_classes}")
if all_classes == {"1"}:
    print("Final check passed: Only one class (1) is present in the annotations.")
else:
    print("Warning: Unexpected class IDs found. Please verify your labels.")
