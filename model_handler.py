from ultralytics import YOLO

# Classes to detect
name_detect = ['bus', 'car', 'motorbike', 'truck']

# Load model
model = YOLO('11s_retrain.pt')
all_names = model.names

# Lấy ID của các lớp đó
ids_to_detect = [id for id, name in all_names.items() if name in name_detect]