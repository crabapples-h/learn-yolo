import time

import cv2
from ultralytics import YOLO

model = YOLO('./runs/detect/train6/weights/best.pt')
# 打开默认摄像头（通常是0）
cap = cv2.VideoCapture(1)
# 检查摄像头是否成功打开
if not cap.isOpened():
    print("无法打开摄像头")
    exit()
while True:
    # 读取一帧
    ret, frame = cap.read()
    # 如果正确读取帧，ret为True
    if not ret:
        print("无法获取帧")
        break
    # 显示帧
    results = model(frame)
    # 在帧上绘制检测结果
    annotated_frame = results[0].plot()
    # 显示结果
    cv2.imshow('YOLO实时检测', annotated_frame)
    # 按'q'键退出
    if cv2.waitKey(1) == ord('q'):
        break
# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()
