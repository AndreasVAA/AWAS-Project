import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuration: Define file paths and toggles directly in the code.
CSV_FILE = "/home/itk/Desktop/Andreas/AWAS-Project/RT_DETR_MODEL/runs/RT_DETR_TESTING/results.csv"  # Path to your DETR training CSV file.
OUTPUT_DIR = "plots"                         # Base folder for saving all plots.
RUN_FOLDER_NAME = "RT_DETR_1280"               # Manually set run-specific folder name; change per run.
PLOT_LOSS = True                             # Set to True to generate loss curves.
PLOT_MAP = True                              # Set to True to generate mAP curves.
PLOT_LR = True                               # Set to True to generate learning rate curves.

def plot_loss_curves(df, save_folder=None):
    """Plot training and validation loss curves for DETR."""
    plt.figure()
    # Plot training losses for GIoU, classification, and L1 losses.
    plt.plot(df['epoch'], df['train/giou_loss'], label='Train GIoU Loss')
    plt.plot(df['epoch'], df['train/cls_loss'], label='Train Cls Loss')
    plt.plot(df['epoch'], df['train/l1_loss'], label='Train L1 Loss')
    # Plot validation losses for GIoU, classification, and L1 losses.
    plt.plot(df['epoch'], df['val/giou_loss'], label='Val GIoU Loss')
    plt.plot(df['epoch'], df['val/cls_loss'], label='Val Cls Loss')
    plt.plot(df['epoch'], df['val/l1_loss'], label='Val L1 Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    if save_folder is not None:
        out_file = os.path.join(save_folder, "loss_curves.png")
        plt.savefig(out_file)
        print(f"Saved loss curves to {out_file}")

def plot_map_curves(df, save_folder=None):
    """Plot mAP50 and mAP50-95 curves over epochs."""
    plt.figure()
    plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP50')
    plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label='mAP50-95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP Curves Over Epochs')
    plt.legend()
    plt.grid(True)
    if save_folder is not None:
        out_file = os.path.join(save_folder, "map_curves.png")
        plt.savefig(out_file)
        print(f"Saved mAP curves to {out_file}")

def plot_learning_rates(df, save_folder=None):
    """Plot learning rate for each parameter group over epochs."""
    plt.figure()
    plt.plot(df['epoch'], df['lr/pg0'], label='LR PG0')
    plt.plot(df['epoch'], df['lr/pg1'], label='LR PG1')
    plt.plot(df['epoch'], df['lr/pg2'], label='LR PG2')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Over Epochs')
    plt.legend()
    plt.grid(True)
    if save_folder is not None:
        out_file = os.path.join(save_folder, "learning_rate_curves.png")
        plt.savefig(out_file)
        print(f"Saved learning rate curves to {out_file}")

def main():
    # Load CSV file into a DataFrame.
    try:
        df = pd.read_csv(CSV_FILE)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    if 'epoch' not in df.columns:
        print("CSV file must contain an 'epoch' column.")
        return

    # Create the base output folder ("plots") if it doesn't exist.
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created base folder: {OUTPUT_DIR}")

    # Create a run-specific subfolder using the manually defined RUN_FOLDER_NAME.
    run_folder = os.path.join(OUTPUT_DIR, RUN_FOLDER_NAME)
    if not os.path.exists(run_folder):
        os.makedirs(run_folder)
        print(f"Created folder for run: {run_folder}")
    else:
        print(f"Using existing folder for run: {run_folder}")

    # Generate and save plots based on toggles.
    if PLOT_LOSS:
        plot_loss_curves(df, save_folder=run_folder)
    if PLOT_MAP:
        plot_map_curves(df, save_folder=run_folder)
    if PLOT_LR:
        plot_learning_rates(df, save_folder=run_folder)

if __name__ == "__main__":
    main()
