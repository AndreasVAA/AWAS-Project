#!/usr/bin/env python3
"""
Recursively increment the class ID in YOLO-format .txt label files by +1,
while reporting counts of instances per class before and after the operation.
Usage:
    python increment_labels.py /path/to/labels_folder
"""
import os
import sys
from collections import Counter

# Counters to track class frequencies before and after incrementing
before_counts = Counter()
after_counts = Counter()


def process_file(path: str):
    """
    Read a label file, increment the first token (class) on each well-formed line,
    and overwrite the file if changes were made. Updates global counters.
    """
    lines_out = []
    changed = False

    with open(path, 'r') as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            # Preserve blank lines
            if not stripped:
                lines_out.append(line)
                continue

            parts = stripped.split()
            # Expect exactly 5 tokens: class xmin ymin xmax ymax
            if len(parts) != 5:
                print(f"⚠️  Skipping malformed line {lineno} in {path!r}: found {len(parts)} tokens")
                lines_out.append(line)
                continue

            # Parse and count the old class
            try:
                cls = int(parts[0])
            except ValueError:
                print(f"⚠️  Non-integer class on line {lineno} in {path!r}: {parts[0]!r}")
                lines_out.append(line)
                continue

            before_counts[cls] += 1
            new_cls = cls + 1
            after_counts[new_cls] += 1

            # Rebuild and store updated line
            new_line = " ".join([str(new_cls)] + parts[1:]) + "\n"
            lines_out.append(new_line)
            changed = True

    # Overwrite only if there was at least one change
    if changed:
        with open(path, 'w') as f:
            f.writelines(lines_out)
        print(f"✔️  Updated classes in {path!r}")
    else:
        print(f"– No class increments needed in {path!r}")


def walk_and_process(root_dir: str):
    """
    Recursively walk root_dir and process every .txt file.
    """
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith('.txt'):
                file_path = os.path.join(dirpath, fn)
                process_file(file_path)


def print_summary():
    """
    Print frequency counts of classes before and after incrementing.
    """
    print("\n=== Summary Before Increment ===")
    print(f"Distinct classes: {len(before_counts)}")
    for cls, cnt in sorted(before_counts.items()):
        print(f"  Class {cls}: {cnt} instances")

    print("\n=== Summary After Increment ===")
    print(f"Distinct classes: {len(after_counts)}")
    for cls, cnt in sorted(after_counts.items()):
        print(f"  Class {cls}: {cnt} instances")


if __name__ == "__main__":
    # Configuration: set your labels folder path here
    ROOT_DIR = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_MULTICLASS_WITHOUT_COPEPOD_IN_USE/train/labels_minmax"
    # Process without CLI arguments
    if not os.path.isdir(ROOT_DIR):
        print(f"Error: {ROOT_DIR!r} is not a directory.")
    else:
        walk_and_process(ROOT_DIR)
        print_summary()
