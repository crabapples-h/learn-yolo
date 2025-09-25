from ultralytics import YOLO

# 加载预训练模型
model = YOLO('models/yolov8x.pt')
model('./resources/目标检测.png', save=True, show=True)
