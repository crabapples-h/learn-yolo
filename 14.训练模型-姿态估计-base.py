from ultralytics import YOLO
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
print("训练完成")
