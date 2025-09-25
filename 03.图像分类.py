from ultralytics import YOLO

# 加载预训练模型
model = YOLO('models/yolov8x.pt')
model.train(data='train.yaml', epochs=200, imgsz=640, batch=32, device='cpu')
