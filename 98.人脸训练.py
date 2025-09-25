from ultralytics import YOLO

# 加载预训练模型
yolo = YOLO('models/yolov8x.pt')
# 训练模型   
# data:数据集路径   
# epochs:训练轮数  
# imgsz: 图片大小   
# batch: 批次大小
# device: 使用设备 0:GPU 'cpu':CPU

yolo.train(data='./dataset/train.yaml', epochs=50, imgsz=640,batch=32,device='cpu')
print("训练完成")