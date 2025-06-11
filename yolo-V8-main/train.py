from ultralytics import YOLO

# Load a model
model = YOLO('ultralytics/models/v8/yolov8n.yaml') # build a new model from scratch

# Use the model
results = model.train(data='UAV.yaml',
                      epochs=300,
                      patience=100,
                      batch=16,
                      device=0,
                      name="yolov8_hardware_v0",
                      optimizer="AdamW",
                      cos_lr=True,
                      lr0=0.01,
                      plots=True
                      )
