from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11m.pt")

# Define search space
search_space = {
    "lr0": (1e-5, 1e-1),
    "degrees": (0.0, 45.0),
}

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="/home/itk/Desktop/Andreas/AWAS-Project/YOLO/dataConf.yaml",
    epochs=30,
    iterations=10,
    optimizer="SGD",
    space=search_space,
    plots=False,
    save=False,
    val=False,
)