from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

results = model.train(
        # Data and model settings
        data="/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf_multiple_Classes.yaml",
        epochs=500,
        batch=2,
        imgsz=(1280, 960),

        # Experiment output
        project="runs",
        name="RT_DETR_multiple_classes",

        # Optimization settings
        optimizer="SGD",
        seed=0,
        pretrained=True,

        # Hardware and performance
        device=0,
        workers=8,
        amp=True,

        # Additional training hyperparameters
        patience=100,
        save=True,
        save_period=-1,
        verbose=True,
        deterministic=True,

        # Data augmentation and scheduling
        single_cls=False,
        rect=False,
        cos_lr=False,
        mosaic=1,
        mixup=0.0,
        auto_augment="randaugment",
        erasing=0.4,
        hsv_h=0.7,
        hsv_s=0.015,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        crop_fraction=1,
                                     
        # Loss function weights and other parameters
        #lr0=0.005,
        #momentum=0.9,
        #weight_decay=0.005,

        # Inference/validation settings
        val=True,
        split="val",

        
    )