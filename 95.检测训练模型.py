from ultralytics import YOLO

# 加载训练好的模型
# best.pt: 最佳模型，适用于生产
# last.pt: 最后一轮训练的模型，适用于继续训练
yolo = YOLO('runs/detect/train4/weights/best.pt')
# yolo('resources/input.mp4', show=True, save=True)
yolo('dataset/val/2.png', show=True, save=True)