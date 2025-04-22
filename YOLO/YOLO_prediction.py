from ultralytics import YOLO
import os

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
    #train_results = train_model()
    #print("Training completed. Results:")
   

    # Example usage for prediction:
    pred_results = predict_model()
    print("Prediction completed. Results:")
    #for result in pred_results:
     #   print(result.summary())
    