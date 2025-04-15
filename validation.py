from ultralytics import YOLO

# Load a model
model = YOLO("yolo11s.pt")  # load an official model
model = YOLO("yolo11s_emotion.pt")  # load a custom model

# Validate the model
metrics = model.val(data ="./dataset/images")  
metrics.box.map  # map50-95
metrics.box.map50  # map50
metrics.box.map75  # map75
metrics.box.maps  # a list contains map50-95 of each category