from ultralytics import SAM
import os
from PIL import Image
import cv2

def predict_model_sam(model_path="sam2.1_b.pt",
                      source="/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images/gunnerus_vertikal_2_ny_126.jpg",
                      bboxes=[
    [822, 109, 1021, 327],
    [410, 517, 695, 648],
    [418, 619, 526, 960],
    [1109, 0, 1280, 220],
    [1090, 0, 1278, 108],
    [575, 344, 697, 602],
    [292, 577, 591, 735]
]
,
                      project="SAM_Output",
                      name="prediction",
                      save=True):
    """
    Run segmentation on the given source image using SAM with a bounding box prompt.
    Optionally saves the output segmentation image(s).

    Args:
        model_path (str): Path or name of the SAM model weights.
        source (str): Path to the input image.
        bboxes (list): List of bounding boxes in [x1, y1, x2, y2] format.
        project (str): Folder to save the output.
        name (str): Name for the output subfolder.
        save (bool): Whether to save the segmentation result.

    Returns:
        results: A list of segmentation results objects returned by SAM.
    """
    # Load SAM model
    model = SAM(model_path)

    model.info()
    image = cv2.imread(source)
    if image is None:
        raise ValueError(f"Failed to load image from {source}")

    # Resize the loaded image
    resized_image = cv2.resize(image, (1280, 960), interpolation=cv2.INTER_LINEAR)
        
    # Run segmentation with bounding box prompts
    results = model(resized_image, bboxes=bboxes)
    
    # Ensure output directory exists
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir, exist_ok=True)
    
    if save:
        if isinstance(results, list):
            for idx, r in enumerate(results):
                if hasattr(r, "save"):
                    # Call save with filename argument
                    r.save(filename=os.path.join(save_dir, f"segmented_{idx}.png"))
                else:
                    try:
                        output_image = Image.fromarray(r.img)
                        output_path = os.path.join(save_dir, f"segmented_{idx}.png")
                        output_image.save(output_path)
                        print(f"Saved segmentation to {output_path}")
                    except Exception as e:
                        print("Could not save segmentation output:", e)
        else:
            if hasattr(results, "save"):
                results.save(filename=os.path.join(save_dir, "segmented.png"))
            else:
                try:
                    output_image = Image.fromarray(results.img)
                    output_path = os.path.join(save_dir, "segmented.png")
                    output_image.save(output_path)
                    print(f"Saved segmentation to {output_path}")
                except Exception as e:
                    print("Could not save segmentation output:", e)
    
    return results

if __name__ == "__main__":
    # Example usage:
    pred_results = predict_model_sam(
        model_path="sam_b.pt",
        project="SAM_Output",
        name="Testing_Random_Image_SAM",
        save=True
    )
    print("Prediction completed. Results:", pred_results)
