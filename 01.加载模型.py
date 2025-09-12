from ultralytics import YOLO

yolo = YOLO('models/yolov8n.pt')
# yolo('./resources/01.png',show=True,save=True)
# yolo('./resources/FHC09594.jpg',show=True,save=True)
yolo('./resources/input.mp4',show=True,save=True)