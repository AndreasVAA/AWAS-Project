
from ultralytics import YOLO
import os

print("Working dir - ", os.getcwd())


model = YOLO("yolo11x.pt")

#results = model.train(data="./dataConf.yaml", epochs=1, batch=16, imgsz=(1280,960), project="runs", name="exp")
#results = model.train(data="./dataConf.yaml", epochs=1, batch=16, imgsz=1280, project="runs", name="exp")

results = model.train(
    # Data and model settings
    data="./dataConf.yaml",       # Path to the dataset config file
    epochs=500,                     # Number of training epochs
    batch=3, #3 Did not work with memory                     # Batch size per iteration (reduce this if you encounter OOM errors)
    imgsz=(1280,960),                   # Input image size for training

    # Experiment output
    project="runs",               # Directory to save runs/experiments
    name="YOLO11_modelType_X_1280x960",                   # Name of this experiment

    # Optimization settings
    optimizer="SGD",              # Optimizer selection (auto will choose, along with lr0 and momentum)
    seed=0,                       # Random seed for reproducibility
    pretrained=True,              # Use pretrained weights

    # Hardware and performance
    device=None,                  # Device to use ('cpu', 'cuda', etc.). None auto-selects CUDA if available.
    workers=8,                    # Number of data loader worker processes
    amp=True,                     # Enable Automatic Mixed Precision training

    # Additional training hyperparameters (with defaults from the trainer)
    patience=100,                 # Early stopping patience
    save=True,                    # Save checkpoints during training
    save_period=-1,               # How often (in epochs) to save a checkpoint (-1 for end-of-training only)
    verbose=True,                 # Verbose logging
    deterministic=True,           # Ensure deterministic training (if possible)
    
    # Data augmentation and scheduling (most defaults shown; change as needed)
    single_cls=True,             # Treat dataset as single-class
    rect=False,                   # Use rectangular training
    cos_lr=False,                 # Use cosine learning rate scheduler
    mosaic=1,  #default 1         # Mosaic augmentation strength
    mixup=0.0,                    # Mixup augmentation strength
    auto_augment="randaugment",   # Auto augmentation strategy
    erasing=0.4, # default 0.4              # Random erasing augmentation probability
    hsv_h = 0.7, #defualt 0.015
    hsv_s = 0.015, #default 0.7
    hsv_v = 0.4, #default = 0.4
    translate = 0.1, #default = 0.1
    scale = 0.5, #defualt = 0.5
    fliplr = 0.5, # default = 0.5
    crop_fraction = 1, # default = 1
                                     

    # Loss function weights and other parameters
    lr0=0.005,                     # Initial learning rate (ignored if optimizer='auto')
    momentum=0.9,               # Momentum (ignored if optimizer='auto')
    weight_decay= 0.005, #0.0005,          # Weight decay coefficient

    # Inference/validation settings
    val=True,                     # Run validation during training
    split="val",                  # Which split to use for validation

    # And more parameters are available – see the Ultralytics documentation for the full list.
)
