from ultralytics import YOLO
import torch
# 加载预训练模型
model = YOLO('models/yolov8n-pose.pt')
# model = YOLO('models/yolo11x-pose.pt')
# model = YOLO('models/yolov8n-pose.pt')
# model = YOLO('runs/pose/train2/weights/last.pt')


# 训练模型   
# data:数据集路径   
# epochs:训练轮数  
# imgsz: 图片大小   
# batch: 批次大小
# device: 使用设备 0:GPU 'cpu':CPU
# workers: 使用的进程数
# verbose: 训练进度条
# resume: 恢复训练
# name: 训练结果保存名称
# plot: 绘制训练曲线
# save: 保存训练结果
# save_period: 每隔多少轮保存一次模型
# save_dir: 训练结果保存路径
# weights: 预训练模型路径
model.info()
model.train(data='./dataset1/train.yaml', epochs=300, imgsz=640, batch=32, device=0)



#
# # 检查MPS可用性
# print(f"MPS available: {torch.backends.mps.is_available()}")
# print(f"MPS built: {torch.backends.mps.is_built()}")
#
# model = YOLO('models/yolov8n-pose.pt')
#
# # MPS优化配置
# model.train( data='./dataset1/train.yaml', epochs=300,
#     imgsz=320,  # M2 Pro上建议减小尺寸
#     batch=8,    # 根据内存调整
#     device='mps',  # 使用Apple Metal Performance Shaders
#     workers=2,  # MPS下建议2个worker
#     patience=50,
#     lr0=0.01,
#     lrf=0.01,
#     momentum=0.9,  # MPS上动量稍小
#     weight_decay=0.0005,
#     warmup_epochs=5.0,
#     box=7.5,
#     pose=1.0,   # 增加姿态损失权重
#     kobj=1.5,
#     save=True,
#     exist_ok=True,
#     verbose=True,
#     amp=False   # MPS上关闭自动混合精度
# )
# print("训练完成")
