import os
import cv2
import shutil
import albumentations as A


# Paths for original validation images and labels (update these to your actual paths)
INPUT_IMAGES_DIR = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/images"
INPUT_LABELS_DIR = "/home/itk/Desktop/Andreas/AWAS-Project/AFTI_PMID_SINGLE_CLASS_TESTING_backup_20250215_134318/val/labels"

# Output root folder for the augmented validation set.
OUTPUT_ROOT = "val_augmetnted_lighitning_conditions"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_ROOT, "images")
OUTPUT_LABELS_DIR = os.path.join(OUTPUT_ROOT, "labels")

# Create the output directories if they don't exist.
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_LABELS_DIR, exist_ok=True)

# Define an Albumentations pipeline for heavy light/color augmentation.
# Only include augmentations affecting brightness, contrast, color, etc.
transform = A.Compose([
    # RandomBrightnessContrast adjusts both brightness and contrast.
    # - p=1.0: Always apply this transformation.
    # - brightness_limit=(0.2, 0.6): A random value is picked from this range and added to 1.0 to get the brightness factor.
    #     • Setting brightness_limit to 0 (or (0, 0)) would mean no brightness change.
    #     • With (0.2, 0.6), brightness is increased by 20% to 60% (i.e., factors between 1.2 and 1.6).
    #     • If you set it to a negative range, e.g. (-0.6, -0.2), it would darken the image by 20% to 60%.
    # - contrast_limit=(0., 0.6): Similar to brightness, a value is added to 1.0 for contrast.
    #     • 0 means no contrast change; (0, 0.6) increases contrast by up to 60%.
    A.RandomBrightnessContrast(
        p=1.0,
        brightness_limit=(-0.3, 0.3),
        contrast_limit=(-0.3, 0.3)
    ),
    
    # HueSaturationValue alters the color properties.
    # - p=0.5: This transformation is applied 50% of the time.
    # - hue_shift_limit=20: Shifts the hue channel by a random value in [-20, 20].
    #     • 0 would mean no hue change.
    # - sat_shift_limit=30: Adjusts saturation by a random value in [-30, 30].
    #     • 0 means saturation remains unchanged.
    # - val_shift_limit=20: Adjusts the value (brightness in HSV space) by a random value in [-20, 20].
    #     • 0 means no change.
    A.HueSaturationValue(
        p=0.5,
        hue_shift_limit=10,
        sat_shift_limit=20,
        val_shift_limit=10
    ),
    
    # RandomGamma applies gamma correction.
    # - p=0.5: This transformation is applied half the time.
    # - gamma_limit=(80, 120): The gamma factor is chosen from [0.8, 1.2] (since 100 corresponds to 1.0).
    #     • A gamma factor of 1 (or (100, 100)) results in no change.
    #     • Values below 1 darken the image; above 1 brighten it.
    A.RandomGamma(
        p=0.3,
        gamma_limit=(80, 120)
    ),
    
    # CLAHE (Contrast Limited Adaptive Histogram Equalization) enhances local contrast.
    # - p=0.5: Applied 50% of the time.
    # - clip_limit=2.0: The threshold for contrast clipping.
    #     • A clip_limit of 0 would disable contrast enhancement.
    # - tile_grid_size=(8, 8): The image is divided into 8x8 tiles.
    #     • Smaller grid sizes (e.g., (1,1)) apply equalization globally, while larger values allow for more localized adjustment.
    A.CLAHE(
        p=0.3,
        clip_limit=2.0,
        tile_grid_size=(8, 8)
    ),
    
    # GaussNoise adds Gaussian noise to simulate sensor noise.
    # - p=0.3: This transformation is applied 30% of the time.
    # - std_range=(0.2, 0.44): Standard deviation of the noise is sampled from this range.
    #     • 0 would mean no noise is added.
    #     • Higher values lead to a noisier image.
    A.GaussNoise(
        p=0.3,
        std_range=(0.2, 0.44)
    ),
    
    # RGBShift randomly shifts the color channels.
    # - p=0.3: Applied 30% of the time.
    # - r_shift_limit, g_shift_limit, b_shift_limit (each set to 20): 
    #     • The channels are shifted by a random amount in the range [-20, 20].
    #     • A value of 0 would mean no color shift.
    A.RGBShift(
        p=0.3,
        r_shift_limit=20,
        g_shift_limit=20,
        b_shift_limit=20
    )
])


# Process each image file in the input images directory.
for img_file in os.listdir(INPUT_IMAGES_DIR):
    if not img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    # Full path to the original image.
    img_path = os.path.join(INPUT_IMAGES_DIR, img_file)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Could not read image: {img_path}")
        continue

    # Apply the augmentation pipeline.
    augmented = transform(image=image)
    aug_image = augmented['image']

    # Save the augmented image in the output images folder.
    output_img_path = os.path.join(OUTPUT_IMAGES_DIR, img_file)
    cv2.imwrite(output_img_path, aug_image)

    # Copy the corresponding label file to the output labels folder.
    label_file = os.path.splitext(img_file)[0] + ".txt"
    input_label_path = os.path.join(INPUT_LABELS_DIR, label_file)
    output_label_path = os.path.join(OUTPUT_LABELS_DIR, label_file)
    if os.path.exists(input_label_path):
        shutil.copy(input_label_path, output_label_path)
    else:
        print(f"Label file not found for {img_file}")

print("Augmented validation set created at:", OUTPUT_ROOT)
