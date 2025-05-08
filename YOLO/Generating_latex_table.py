import os
import pandas as pd

# Function to extract the model name from the folder name
def extract_model_name(folder_name):
    model_name = folder_name.split('_')[0]  # Extract model version (e.g., YOLO11n, YOLO11s)
    return model_name

# Function to extract performance metrics from validation_metrics.csv
def extract_performance_metrics(folder_path):
    metrics_file = os.path.join(folder_path, 'validation_metrics.csv')
    df = pd.read_csv(metrics_file)

    metrics = {}
    for index, row in df.iterrows():
        metrics[row['Metric']] = row['Value']
    
    precision = round(metrics['Precision'], 4)
    recall = round(metrics['Recall'], 4)
    f1 = round(metrics['F1'], 4)
    map50 = round(metrics['mAP50'], 4)
    map75 = round(metrics['mAP75'], 4)
    map50_95 = round(metrics['mAP50-95'], 4)

    # Debugging: Print extracted performance metrics
    print(f"Extracted performance metrics for {folder_path}:")
    print(f"  Precision: {precision}, Recall: {recall}, F1: {f1}, mAP50: {map50}, mAP75: {map75}, mAP50-95: {map50_95}")
    
    return precision, recall, f1, map50, map75, map50_95

# Function to extract FPS from the CSV (Core FPS)
def extract_fps(folder_path):
    metrics_file = os.path.join(folder_path, 'validation_metrics.csv')
    df = pd.read_csv(metrics_file)
    fps = df[df['Metric'] == 'Core FPS']['Value'].values[0]
    
    # Debugging: Print extracted FPS
    print(f"Extracted FPS for {folder_path}: {fps}")
    
    return round(fps, 2)

# Function to extract Core Avg Time per Image (ms) from validation_metrics.csv
def extract_avg_inference_time(folder_path):
    metrics_file = os.path.join(folder_path, 'validation_metrics.csv')
    df = pd.read_csv(metrics_file)
    avg_inference_time = df[df['Metric'] == 'Core Avg Time per Image (ms)']['Value'].values[0]
    
    # Debugging: Print extracted Avg Inference Time
    print(f"Extracted Avg Inference Time for {folder_path}: {avg_inference_time} ms")
    
    return round(avg_inference_time, 2)

# Function to extract FLOPs and parameters (hardcoded values)
def extract_flops_and_params(model_name):
    flops_params_dict = {
        'YOLO11n': {'flops': 6.5, 'params': 2.6}, 
        'YOLO11s': {'flops': 21.5, 'params': 9.4},
        'YOLO11m': {'flops': 68.0, 'params': 20.1},
        'YOLO11l': {'flops': 86.9, 'params': 25.3},
        'YOLO11x': {'flops': 194.9, 'params': 56.9}
    }
    
    # Debugging: Print extracted FLOPs and parameters
    print(f"Extracted FLOPs and Params for {model_name}: FLOPs = {flops_params_dict[model_name]['flops']}, Params = {flops_params_dict[model_name]['params']}")
    
    return flops_params_dict.get(model_name, None)

# Main function to generate LaTeX tables
def generate_latex_tables(root_folder, output_file=None):
    # Define memory usage for each model (pre-defined)
    memory_dict = {
        'YOLO11n': 71.94,
        'YOLO11s': 123.92,
        'YOLO11m': 216.78,
        'YOLO11l': 250.07,
        'YOLO11x': 428.39
    }

    performance_table = []
    computational_table = []
    
    all_fps = []
    all_memory = []
    all_flops = []
    all_inference_time = []
    
    # Iterate through the models in the root folder
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        
        # Only process directories (model directories)
        if os.path.isdir(subfolder_path):
            print(f"Processing model directory: {subfolder_path}")
            model_name = extract_model_name(subfolder)
            
            # Extract performance metrics for the model
            precision, recall, f1, map50, map75, map50_95 = extract_performance_metrics(subfolder_path)
            performance_data = [precision, recall, f1, map50, map75, map50_95]
            
            # Extract computational metrics for the model
            flops_and_params = extract_flops_and_params(model_name)
            flops = round(flops_and_params['flops'], 2)
            params = round(flops_and_params['params'], 2)
            memory_usage = memory_dict.get(model_name, None)
            inference_speed = extract_fps(subfolder_path)  # Core FPS
            avg_inference_time = extract_avg_inference_time(subfolder_path)  # Core Avg Time per Image (ms)
            
            computational_data = [flops, inference_speed, avg_inference_time, memory_usage, params]
            
            # Collect data for column-wise highlight (max/min)
            all_fps.append(inference_speed)
            all_memory.append(memory_usage)
            all_flops.append(flops)
            all_inference_time.append(avg_inference_time)
            
            performance_table.append(performance_data)
            computational_table.append(computational_data)
    
    # Prepare LaTeX for Performance Table
    performance_latex = "\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n"
    performance_latex += "Model & Precision & Recall & F1 & mAP50 & mAP75 & mAP50-95 \\\\\n\\hline\n"
    
    for i, row in enumerate(performance_table):
        # Print to verify the order of model data assignment
        print(f"Adding performance data for {list(memory_dict.keys())[i]}: {row}")
        
        # Properly format the performance data for LaTeX
        performance_latex += f"{list(memory_dict.keys())[i]} & " + " & ".join(map(str, row)) + " \\\\\n"
    
    performance_latex += "\\hline\n\\end{tabular}\n"
    
    # Prepare LaTeX for Computational Table
    computational_latex = "\\begin{tabular}{|c|c|c|c|c|c|}\n\\hline\n"
    computational_latex += "Model & FPS & Avg Inference Time (ms) & Mem Usage (MB) & FLOPs & Parameters (M) \\\\\n\\hline\n"
    
    for i, row in enumerate(computational_table):
        # Print to verify the order of model data assignment
        print(f"Adding computational data for {list(memory_dict.keys())[i]}: {row}")
        
        raw_row_data = row.copy()
        raw_row_data[1] = float(raw_row_data[1])  # FPS
        raw_row_data[2] = float(raw_row_data[2])  # Avg Inference Time
        raw_row_data[3] = float(raw_row_data[3])  # Mem Usage (MB)
        raw_row_data[4] = float(raw_row_data[4])  # FLOPs
        
        # Apply bold formatting for max/min values per column
        row_data = row.copy()
        row_data[1] = f"\\textbf{{{row_data[1]}}}" if raw_row_data[1] == max(all_fps) else str(row_data[1])
        row_data[2] = f"\\textbf{{{row_data[2]}}}" if raw_row_data[2] == min(all_inference_time) else str(row_data[2])
        row_data[3] = f"\\textbf{{{row_data[3]}}}" if raw_row_data[3] == min(all_memory) else str(row_data[3])
        row_data[4] = f"\\textbf{{{row_data[4]}}}" if raw_row_data[4] == min(all_flops) else str(row_data[4])
        
        computational_latex += f"{list(memory_dict.keys())[i]} & " + " & ".join(map(str, row_data)) + " \\\\\n"
    
    computational_latex += "\\hline\n\\end{tabular}\n"
    
    # Print LaTeX tables to console
    print("Performance Metrics Table:")
    print(performance_latex)
    
    print("Computational Metrics Table:")
    print(computational_latex)
    
    # Optionally save the LaTeX code to a .tex file
    if output_file:
        with open(output_file, 'w') as file:
            file.write(performance_latex + "\n\n" + computational_latex)

# Define the root folder for model subfolders
root_folder = '/home/andreas/AWAS/AWAS-Project/YOLO/Validation_Batch4_resolution640_ALL_YOLO_MODELS'

# Set the desired output file path for the LaTeX code
output_file = 'latex_output.tex'  # You can modify the file path as needed

# Generate the LaTeX tables
generate_latex_tables(root_folder, output_file)
