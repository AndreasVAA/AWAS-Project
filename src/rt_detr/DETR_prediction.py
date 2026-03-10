from ultralytics import RTDETR
import os
import csv
import time

def run_and_process_prediction(model_path, source_folder, save_dir, imgsz=1280, device="cuda", conf_threshold=0.4, **kwargs):
    """
    Runs prediction using the RTDETR model on a folder of images,
    saves the prediction outputs (images and txt files), and computes basic statistics.
    
    Parameters:
        model_path (str): Path to the RTDETR model weights.
        source_folder (str): Folder containing input images.
        save_dir (str): Folder to save prediction outputs and statistics.
        imgsz (int or tuple): Image size for inference.
        device (str): Device for inference ("cuda" or "cpu").
        conf_threshold (float): Confidence threshold used for predictions.
        **kwargs: Additional keyword arguments for model.predict.
        
    Returns:
        results: The prediction results from the model.
    """
    # Initialize the RTDETR model
    model = RTDETR(model_path)
    
    # Run prediction on the folder of images.
    # Setting save=True and save_txt=True will save images and corresponding txt files with predictions.
    results = model.predict(source=source_folder, imgsz=imgsz, device=device,
                            save=True, save_txt=True, project=save_dir, conf=conf_threshold, **kwargs)
    
    # Compute basic statistics
    total_predictions = 0
    total_confidence = 0.0
    num_images = len(results)  # assuming one result per image
    
    for res in results:
        # Check if predictions exist and compute metrics
        if hasattr(res, 'boxes') and res.boxes is not None:
            boxes = res.boxes
            count = len(boxes)
            total_predictions += count
            if count > 0 and hasattr(boxes, "conf"):
                total_confidence += boxes.conf.mean().item()
    
    avg_confidence = total_confidence / num_images if num_images > 0 else 0
    
    # Save statistics to a CSV file in the save directory
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "prediction_stats.csv")
    with open(csv_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Images", num_images])
        writer.writerow(["Total Predictions", total_predictions])
        writer.writerow(["Average Confidence", avg_confidence])
        writer.writerow(["Confidence Threshold", conf_threshold])
    
    print(f"Predictions and statistics saved to folder: {save_dir}")
    print(f"Statistics CSV: {csv_path}")
    
    return results

if __name__ == "__main__":
    model_path = "/home/itk/Desktop/Andreas/AWAS-Project/RT_DETR_MODEL/runs/RT_DETR_TESTING/weights/best.pt"
    image_source = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/images"
    save_folder = "predictions_rt_detr"
    
    start_time = time.time()
    pred_results = run_and_process_prediction(model_path, image_source, save_folder, imgsz=1280, device="cuda", conf_threshold=0.4)
    total_time = time.time() - start_time
    print(f"Total inference time: {total_time:.2f} seconds")
