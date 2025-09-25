from ultralytics import YOLO

# 加载预训练模型
model = YOLO('models/yolov8n-cls.pt')
results = model('./resources/图像分类.png',save=True)
for result in results:
    probs = result.probs
    print(probs)
