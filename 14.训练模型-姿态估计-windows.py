from ultralytics import YOLO
import torch

# 检查GPU信息
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
print(f"当前GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024 ** 3:.1f} GB")

model = YOLO('yolov8n-pose.pt')

# 3080优化配置
model.train(
    data='./dataset/train.yaml',
    epochs=300,
    imgsz=640,  # 3080可以处理原尺寸
    batch=8,  # 根据12GB内存调整
    device=0,  # 使用GPU 0
    workers=8,  # 充分利用CPU核心
    patience=50,
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,

    # # 重点：大幅调整损失权重
    # pose=10.0,  # 大幅提高姿态损失权重
    # kobj=5.0,  # 提高关键点目标权重
    # box=1.0,  # 降低检测权重（因为检测已经很好）
    # cls=0.3,  # 降低分类权重


    # 性能优化
    amp=True,  # 自动混合精度训练
    cos_lr=True,  # 余弦学习率调度
    close_mosaic=10,  # 最后10epoch关闭马赛克增强

    # 数据增强
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,  # 水平翻转对姿态很重要

    save=True,
    exist_ok=True,
    verbose=True
)
print("训练完成")
