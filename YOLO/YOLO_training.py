from ultralytics import YOLO
import os

def train_model(model_path="yolo11m.pt", config_path="./dataConf_multiple_Classes.yaml"):
    """
    Train the YOLO model using the specified configuration.
    Allows overriding default parameters via keyword arguments.
    """
    print("Working dir -", os.getcwd())
    model = YOLO(model_path, task="detect")
    results = model.train(
        # Data and model settings
        data=config_path,
        epochs=1000,
        batch=6,
        imgsz=960,

        # Experiment output
        project="runs_with_mutiple_classes",
        name="YOLO11M_960x960",

        # Optimization settings
        optimizer="auto",
        seed=0,
        pretrained=True,

        # Hardware and performance
        device=None,
        workers=8,
        amp=True,

        # Additional training hyperparameters
        patience=100,
        save=True,
        save_period=-1,
        verbose=False,
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
        lr0=0.005,
        momentum=0.9,
        weight_decay=0.005,

        # Inference/validation settings
        val=True,
        split="val",
        
    )
    return results

def predict_model(model_path = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/runs/YOLO11_modelType_M_1280x960/weights/best.pt",
                  source = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images/gunnerus_vertikal_2_ny_126.jpg",
                  imgsz=(1280, 960),
                  device="cuda",
                  visualize=True,
                  project="Interference",
                  name="DUM_Testing",
                  show=True,
                  save=True,
                  save_txt=True,
                  save_conf=True,
                  save_crop=True,
                  show_conf=True):
    """
    Run inference on the given source (image path, video, etc.) using the YOLO model.
    Returns a list of Results objects.
    """
    model = YOLO(model_path, "detect")
    results = model(source,
                    imgsz=imgsz,
                    device=device,
                    #visualize=visualize,
                    project=project,
                    name=name,
                    #show=show,
                    save=save,
                    save_txt=save_txt,
                    save_conf=save_conf,
                    save_crop=save_crop,
                    show_conf=show_conf)
    return results

if __name__ == "__main__":
    # Example usage for training:
    train_results = train_model()
    print("Training completed. Results:", train_results)

    # Example usage for prediction:
    #pred_results = predict_model()
    #print("Prediction completed. Results:", pred_results)
