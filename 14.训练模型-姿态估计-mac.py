from ultralytics import YOLO
import torch

# 加载预训练模型
model = YOLO('models/yolov8n-pose.pt')
# model = YOLO('models/yolo11x-pose.pt')
# model = YOLO('models/yolov8n-pose.pt')
# model = YOLO('runs/pose/train2/weights/last.pt')

model.info()
# 检查MPS可用性
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
# MPS优化配置
model.train(data='./dataset1/train.yaml', epochs=300,
            imgsz=320,  # M2 Pro上建议减小尺寸
            batch=8,  # 根据内存调整
            device='mps',  # 使用Apple Metal Performance Shaders
            workers=2,  # MPS下建议2个worker
            patience=50,
            lr0=0.01,
            lrf=0.01,
            momentum=0.9,  # MPS上动量稍小
            weight_decay=0.0005,
            warmup_epochs=5.0,
            box=7.5,
            pose=1.0,  # 增加姿态损失权重
            kobj=1.5,
            save=True,
            exist_ok=True,
            verbose=True,
            amp=False  # MPS上关闭自动混合精度
            )
print("训练完成")
