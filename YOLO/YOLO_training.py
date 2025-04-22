from ultralytics import YOLO
import os

def train_model(model_path="yolo11m.pt", config_path="./dataConf.yaml"):
    """
    Train the YOLO model using the specified configuration.
    Allows overriding default parameters via keyword arguments.
    """
    print("Working dir -", os.getcwd())
    model = YOLO(model_path, task="detect")
    results = model.train(
        # Data and model settings
        data=config_path,
        epochs=1500,
        batch=3,
        imgsz= (1280,960),

        # Experiment output
        project="rDifferent_optimizers_tetsing",
        name="YOLO11M_1280_ADAM_default_learning_rates_longer_training",

        # Overwrite folder
        exist_ok=True,

        # Optimization settings
        optimizer="adam",
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
        hsv_h=0.7, # Default is 0.7
        hsv_s = 0.015, # Default is 0.015
        hsv_v=0.4, # Defaut is 0.4
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
    return results


if __name__ == "__main__":
    # Example usage for training:
    train_results = train_model()
    print("Training completed.")
   

    # Example usage for prediction:
    #pred_results = predict_model()
    #print("Prediction completed. Results:")
    #for result in pred_results:
     #   print(result.summary())
    
