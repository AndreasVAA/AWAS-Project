# AWAS — Thesis Project

This project focuses on developing a deep learning-based system to classify phytoplankton species from RGB and hyperspectral images. By leveraging CNNs, transfer learning, and multi-modal approaches, the aim is to improve real-time species identification for autonomous marine monitoring.

Three object detection models are compared: **YOLO11**, **Faster R-CNN**, and **RT-DETR**, evaluating performance across single-class and multi-class scenarios at various input resolutions (640, 960, 1280).

---

## Repository Structure

```
AWAS-Project/
│
├── src/                        # Source code (all Python scripts)
│   ├── yolo/                   #   YOLO training, validation, prediction & plotting
│   │   └── tuning/             #   Hyperparameter tuning scripts (Ray / Ultralytics)
│   ├── faster_rcnn/            #   Faster R-CNN training, validation & utilities
│   ├── rt_detr/                #   RT-DETR training, validation & prediction
│   ├── data_processing/        #   Data augmentation & light-stress generation
│   ├── cropping/               #   Image cropping utilities
│   │   ├── padding/            #     Padding-based cropping
│   │   └── sam/                #     Segment Anything Model (SAM) cropping
│   ├── pipeline/               #   End-to-end detection + cropping pipeline
│   └── external_testing/       #   Inference on external datasets (NALIA)
│
├── configs/                    # YAML configuration files for data paths & classes
│   ├── yolo_single_class.yaml
│   ├── yolo_multiple_classes.yaml
│   ├── data_processing_single_class.yaml
│   ├── data_processing_light_augmented.yaml
│   └── nalia_data.yaml
│
├── data/                       # Datasets
│   └── single_class_dataset/   #   Single-class plankton dataset (train/val splits)
│       ├── train/
│       │   ├── images/
│       │   ├── labels/
│       │   └── labels_minmax/
│       └── val/
│           ├── images/
│           ├── labels/
│           └── labels_minmax/
│
├── outputs/                    # Training runs, validation results & model outputs
│   ├── yolo/                   #   YOLO experiment outputs (runs, validations, plots)
│   ├── faster_rcnn/            #   Faster R-CNN experiment outputs
│   ├── rt_detr/                #   RT-DETR experiment outputs
│   ├── data_processing/        #   Augmented data & interference test results
│   ├── cropping/               #   Cropping evaluation results
│   │   ├── padding/
│   │   └── sam/
│   └── external_testing/       #   NALIA dataset inference results
│
├── docs/                       # Documentation & notes
│   ├── Notes_to_self.txt       #   Development notes and progress tracking
│   └── latex_output.tex        #   LaTeX-formatted result tables for thesis
│
├── .gitignore
└── README.md
```

### Quick Guide

| Looking for…                  | Go to…                |
|-------------------------------|-----------------------|
| Python source code            | `src/`                |
| Data config files (YAML)      | `configs/`            |
| Training/validation datasets  | `data/`               |
| Experiment results & outputs  | `outputs/`            |
| Documentation & thesis notes  | `docs/`               |

---

## Models

| Model        | Framework     | Scripts Location   |
|--------------|---------------|--------------------|
| YOLO11       | Ultralytics   | `src/yolo/`        |
| Faster R-CNN | PyTorch       | `src/faster_rcnn/` |
| RT-DETR      | Ultralytics   | `src/rt_detr/`     |

## Key Dependencies

- Python 3
- PyTorch / torchvision
- Ultralytics (YOLO, RT-DETR)
- OpenCV (cv2)
- Albumentations
- NumPy
- Matplotlib

## Note on Paths

The Python scripts and YAML configs contain hardcoded absolute paths from the original development machine (`/home/itk/Desktop/Andreas/...`). Update these paths to match your local environment before running any scripts.
