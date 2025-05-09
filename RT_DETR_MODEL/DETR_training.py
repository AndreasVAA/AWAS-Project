from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model

import os

def train_model(lr0 = 0.01, rect=False , model_path ="rtdetr-l.pt", project= "DUMDUMTESTING", optimizer = "auto", config_path="/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml", imgsz=640, batch=16, epochs=500, name="Dum_DUm_testing", freeze=None):    
    """
    Train the RTDETR model using the specified configuration.
    Allows overriding default parameters via keyword arguments.
    """
    model = RTDETR(model_path)
    print("Working dir -", os.getcwd())
    results = model.train(
        # Data and model settings
        data=config_path,
        epochs=epochs,
        batch=batch,
        imgsz= imgsz,
        freeze=freeze,
        rect=rect,

        # Experiment output
        project=project,
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
        patience=75,
        save=True,
        save_period=-1,
        verbose=False,
        deterministic=True,

        # Data augmentation and scheduling
        single_cls=False,
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
        #lr0=lr0,
        #momentum=0.9,
        #weight_decay=0.005,

        # Inference/validation settings
        val=True,
        split="val",
        
    )
    return results



if __name__ == "__main__":
    # Example usage for training:
    #Need to run - hvis det ikke går - lavere batch size til 4

    project = "Testing_RT_DETR_variations_longer_patience"
    
    train_results = train_model(imgsz=1024, name="RT_DETR_1280_batch4", batch=4, project=project)
    train_results = train_model(imgsz=640, name="RT_DETR_640_batch4", batch=4, project=project)
    train_results = train_model(imgsz=960, name="RT_DETR_960_batch4", batch=4, project=project)
    
    
    
    
    
   
   
    #Might wnat to run and maybe at 1024 as well
    #train_results = train_model(imgsz=960, name="YOLO11m_960_batch4", batch=4, model_path="yolo11m.pt")
   

    #backbone of YOLO 0-9
    #Neck: 10-22
    #Detect/Head: 23
    # 1. Load YOLO11x
    """

    model_11m = YOLO('yolo11m.pt')

    layers11 = model_11m.model.model  # a nn.Sequential of 24 modules
    print("YOLO11m modules (index: module):")
    for idx, layer in enumerate(layers11):
        print(f"{idx:3d}:", layer)

    
    model_11x = YOLO('yolo11x.pt')

    layers11 = model_11x.model.model  # a nn.Sequential of 24 modules
    print("YOLO11x modules (index: module):")
    for idx, layer in enumerate(layers11):
        print(f"{idx:3d}:", layer)

    

    model_v5x = YOLO('yolov5x.pt')

    layersv5 = model_v5x.model.model  # a nn.Sequential of 24 modules
    print("YOLO5vx modules (index: module):")
    for idx, layer in enumerate(layersv5):
        print(f"{idx:3d}:", layer)
   
    
    """
    


    print("Training completed.")
   

    # Example usage for prediction:
    #pred_results = predict_model()
    #print("Prediction completed. Results:")
    #for result in pred_results:
     #   print(result.summary())
    
