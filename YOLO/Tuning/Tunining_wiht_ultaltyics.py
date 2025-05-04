import ultralytics as ultralytics

from ultralytics import YOLO

import numpy as np






def custom_fitness(self):

    # Custom fitness function

    w = [0.15, 0.25, 0.3, 0.3]  # weights for [P, R, mAP@0.5, mAP@0.5:0.95]

    return (np.array(self.mean_results()) * w).sum()



ultralytics.utils.metrics.fitness = custom_fitness





# Initialize the YOLO model


model = YOLO("yolo11m.pt", task= "detect")


# Define search space

search_space = {
    
    "lr0":             (1e-4, 1e-1),       # range [1e-5, 1e-1]; default: 0.01 (SGD), 0.001 (Adam/AdamW)

    "lrf":             (1e-4, 0.1),        # range [1e-4, 0.1]; default: 0.01

    "momentum":        (0.6, 0.98),        # range (0.6, 0.98); default: 0.937

    "weight_decay":    (0.0, 0.001),       # range [0.0, 0.001]; default: 0.0005

    "warmup_epochs":   (0.0, 5.0),         # range [0.0, 5.0]; default: 3.0

    "warmup_momentum": (0.0, 0.95),        # range [0.0, 0.95]; default: 0.8

    "box":             (1.0, 20.0),        # range [1.0, 20.0]; default: 7.5

    "cls":             (0.2, 4.0),         # range [0.2, 4.0]; default: 0.5

    "dfl":             (0.4, 6.0),         # range [0.4, 6.0]; default: 1.5

    "hsv_h":           (0.0, 0.1),         # range [0.0, 0.1]; default: 0.015

    "hsv_s":           (0.0, 0.9),         # range [0.0, 0.9]; default: 0.7

    "hsv_v":           (0.0, 0.9),         # range [0.0, 0.9]; default: 0.4

    "degrees":         (0.0, 45.0),        # range [0.0, 45.0]; default: 0.0

    "translate":       (0.0, 0.9),         # range [0.0, 0.9]; default: 0.1

    "scale":           (0.0, 0.95),        # range [0.0, 0.95); default: 0.5

    "shear":           (0.0, 10.0),        # range [0, 10]; default: 0.0

    "perspective":     (0.0, 0.001),       # range [0.0, 0.001]; default: 0.0

    "flipud":          (0.0, 1.0),         # range [0.0, 1.0]; default: 0.0

    "fliplr":          (0.0, 1.0),         # range [0.0, 1.0]; default: 0.5

    "bgr":             (0.0, 1.0),         # range [0.0, 1.0]; default: 0.0

    "mosaic":          (0.0, 1.0),         # range [0.0, 1.0]; default: 1.0

    "mixup":           (0.0, 1.0),         # range [0.0, 1.0]; default: 0.0

    "copy_paste":      (0.0, 1.0),         # range [0.0, 1.0]; default: 0.0

    
}


# Tune hyperparameters on COCO8 for 30 epochs

model.tune(

    data="/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml",

    epochs=20,

    iterations=200,

    optimizer="SGD",

    space=search_space,

    plots=False,

    save=True,

    val=False,

    project = "Coarse_tuning_with_ultaltyics",

    workers=8,

    imgsz=1280,

    exist_ok=True,
    
    batch=3,

)
