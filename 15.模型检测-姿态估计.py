from ultralytics import YOLO

# 加载训练好的模型
# best.pt: 最佳模型，适用于生产
# last.pt: 最后一轮训练的模型，适用于继续训练
yolo = YOLO('runs/pose/train2/weights/best.pt')
# yolo = YOLO('models/yolo11n-pose.pt')
# yolo('resources/input.mp4', show=True, save=True)
yolo('resources/姿态估计-自定义', show=True, save=True)