from ultralytics import YOLO

# 加载训练好的模型
# best.pt: 最佳模型，适用于生产
# last.pt: 最后一轮训练的模型，适用于继续训练
# model = YOLO('runs/pose/train/weights/best.pt')
model = YOLO('models/best.pt')
# model = YOLO('models/yolo11n-pose.pt')
# model('resources/input.mp4', show=True, save=True)
# model.predict('resources/姿态估计-自定义', show=False, save=True)
model.predict('dataset3/train/images', show=False, save=True)