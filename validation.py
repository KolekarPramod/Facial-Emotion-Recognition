from ultralytics import YOLO


model = YOLO("yolo11s_emotion.pt")  # load a custom model

# Validate the model
metrics = model.val(data ="./dataset/images/train_split")  
