from ultralytics import YOLO
import os

def train_model(optimizer = "auto", model_path="yolo11n.pt", config_path="/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml", imgsz=640, batch=16, epochs=500, name="Dum_DUm_testing"):    
    """
    Train the YOLO model using the specified configuration.
    Allows overriding default parameters via keyword arguments.
    """
    print("Working dir -", os.getcwd())
    model = YOLO(model_path, task="detect")
    results = model.train(
        # Data and model settings
        data=config_path,
        epochs=epochs,
        batch=batch,
        imgsz= imgsz,

        # Experiment output
        project="Testing_batch_resolution_variations_single_class",
        name=name,

        # Overwrite folder
        exist_ok=True,

        # Optimization settings
        optimizer=optimizer,
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

def train_model_custom_lr(lr, lrf, optimizer = "auto", model_path="yolo11n.pt", config_path="/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml", imgsz=640, batch=16, epochs=500, name="Dum_DUm_testing"):    
    """
    Train the YOLO model using the specified configuration.
    Allows overriding default parameters via keyword arguments.
    """
    print("Working dir -", os.getcwd())
    model = YOLO(model_path, task="detect")
    results = model.train(
        # Data and model settings
        data=config_path,
        epochs=epochs,
        batch=batch,
        imgsz= imgsz,

        # Experiment output
        project="Testing_batch_resolution_variations_single_class",
        name=name,

        # Overwrite folder
        exist_ok=True,

        # Optimization settings
        optimizer=optimizer,
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
        lr0=lr,
        lrf=lrf,
        #momentum=0.9,
        #weight_decay=0.005,

        # Inference/validation settings
        val=True,
        split="val",
        
    )
    return results


if __name__ == "__main__":
    # Example usage for training:
    train_results = train_model(imgsz=1280, name="YOLO11n_1280_batch8", batch=8)
    train_results = train_model(imgsz=1024, name="YOLO11n_1024_batch16")
    train_results = train_model_custom_lr(lr= 0.003, lrf= 0.0005, imgsz=1280, model_path="yolo11m.pt", name="YOLO11m_1280_batch3_SGD_lr_lrf_custom", batch=3, epochs=1000, optimizer="SGD")
    
    print("Training completed.")
   

    # Example usage for prediction:
    #pred_results = predict_model()
    #print("Prediction completed. Results:")
    #for result in pred_results:
     #   print(result.summary())
    
