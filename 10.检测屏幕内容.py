from ultralytics import YOLO
import cv2
import pyautogui
import numpy as np



# 加载训练好的模型
# best.pt: 最佳模型，适用于生产
# last.pt: 最后一轮训练的模型，适用于继续训练
yolo = YOLO('runs/detect/train7/weights/best.pt')

# 指定屏幕范围
# x,y,width,height  全屏None
window = None
while True:
    # 获取屏幕截图
    screenshot = pyautogui.screenshot()
    frame = np.array(screenshot)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    
    # 使用YOLO检测
    results = yolo(frame)
    
    # 显示结果
    annotated_frame = results[0].plot()
    cv2.imshow("Screen Detection", annotated_frame)
    
    # 按'q'退出
    if cv2.waitKey(1) == ord('q'):
        break



cv2.destroyAllWindows()