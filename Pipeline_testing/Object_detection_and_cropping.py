from ultralytics import YOLO
import os

def predict_with_model(model_path = "/home/itk/Desktop/Andreas/AWAS-Project/YOLO/runs/YOLO11_modelType_M_1280x960/weights/best.pt",
                  source = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/train/images/gunnerus_vertikal_2_ny_126.jpg",
                  imgsz=(1280, 960),
                  device="cuda",
                  visualize=True,
                  project="Interference",
                  name="Testing_Image_when cosndiering_cropping_input",
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



def run_evaluation(image_folder, label_folder, output_dir, num_images=4,
                   global_target_size=(1280,960), classification_target_size=(224,224),
                   fixed_pad_ratio=0.125, fill_mode='mean', conf_threshold=0.85):
    """
    Run evaluation mode on a random sample of num_images from image_folder.
    In production, if there are any images whose filenames start with a letter,
    3 out of the selected images will be chosen from that group (if available),
    and the remaining from other images.
    """
    img_extensions = ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    for ext in img_extensions:
        image_files.extend(glob.glob(os.path.join(image_folder, ext)))
    if not image_files:
        print("No images found in", image_folder)
        return

    # Partition images into those starting with a letter and those starting with a digit.
    letter_images = [img for img in image_files if os.path.basename(img)[0].isalpha()]
    digit_images = [img for img in image_files if os.path.basename(img)[0].isdigit()]

    selected_images = []
    if letter_images:
        num_letter = min(3, len(letter_images))
        selected_images.extend(random.sample(letter_images, num_letter))
        remaining = num_images - num_letter
        # If there are digit images, choose from them; otherwise, fill with letter images.
        if remaining > 0:
            if digit_images:
                selected_images.extend(random.sample(digit_images, min(remaining, len(digit_images))))
            else:
                selected_images.extend(random.sample(letter_images, min(remaining, len(letter_images))))
    else:
        # If no letter images, choose randomly from all images.
        selected_images = random.sample(image_files, min(num_images, len(image_files)))
    
    for img_file in selected_images:
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        eval_folder = os.path.join(output_dir, base_name, "evaluation")
        os.makedirs(eval_folder, exist_ok=True)
        evaluate_image(img_file, label_folder, eval_folder,
                       global_target_size, classification_target_size,
                       fixed_pad_ratio, fill_mode, conf_threshold)



if __name__ == "__main__":
    predicted_results = predict_with_model

