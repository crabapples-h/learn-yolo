from ultralytics import YOLO
# 文档https://docs.ultralytics.com/zh/
# model = YOLO('models/yolov8n.pt')
model = YOLO('models/yolov8x.pt')
# model('./resources/01.png',show=True,save=True)
# model('./resources/FHC09594.jpg',show=True,save=True)
# model('./resources/02.jpg',show=True,save=True)
model('./dataset/val',save=True)