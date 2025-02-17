
from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.train(data="YOLO/dataset.yaml", epochs=1, batch=16, imgsz=(1280,960), project="runs/testing", name="exp")