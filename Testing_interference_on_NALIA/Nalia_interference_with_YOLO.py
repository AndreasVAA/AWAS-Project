import json
from ultralytics import YOLO
import time

# Load model
model = YOLO("/home/itk/Desktop/Andreas/AWAS-Project/YOLO/runs/YOLO11_modelType_M_1280x960/weights/best.pt")

# Measure inference time
start_time = time.time()

# Run test
results = model.val(data="/home/itk/Desktop/Andreas/AWAS-Project/Testing_interference_on_NALIA/Nalia_data.yaml", batch = 3, imgsz = (1280,960))

end_time = time.time()
inference_time = end_time - start_time
fps = len(results) / inference_time

# Extract relevant information from results
structured_results = []
for result in results:
    structured_result = {
        'image': result.path,
        'boxes': [],
        'scores': [],
        'classes': []
    }
    for box in result.boxes:
        structured_result['boxes'].append(box.xyxy.tolist())
        structured_result['scores'].append(box.conf.tolist())
        structured_result['classes'].append(box.cls.tolist())
    structured_results.append(structured_result)

# Extract summary metrics
summary_metrics = {
    'precision': results.metrics.precision,
    'recall': results.metrics.recall,
    'mAP50': results.metrics.map50,
    'mAP50-95': results.metrics.map50_95,
    'mAP': results.metrics.map,
    'TP': results.metrics.tp,
    'FP': results.metrics.fp,
    'FN': results.metrics.fn,
    'inference_time': inference_time,
    'fps': fps
}

# Save structured results to a JSON file in a specific directory
output_path = '/home/itk/Desktop/Andreas/AWAS-Project/Testing_interference_on_NALIA/structured_results.json'
with open(output_path, 'w') as json_file:
    json.dump({
        'results': structured_results,
        'summary': summary_metrics
    }, json_file, indent=4)

print(f"Results and summary saved to {output_path}")