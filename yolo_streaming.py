from ultralytics import solutions

inf = solutions.Inference(
    model="yolo11n.pt",  # you can use any model that Ultralytics support, i.e. YOLO11, YOLOv10
)

inf.inference()