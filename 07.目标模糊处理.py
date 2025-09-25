import cv2
from ultralytics import YOLO

# 加载预训练的YOLOv8模型
model = YOLO('models/yolo11n.pt')
# 打印模型能够识别的所有物体类别名称
print(model.names)
# 初始化摄像头，参数1表示使用默认的第二个摄像头（如果只有一个摄像头，通常使用0）
cap = cv2.VideoCapture(1)
# 循环读取摄像头帧，直到摄像头关闭
while cap.isOpened():
    # 读取一帧图像
    ret, frame = cap.read()
    # 如果没有成功读取帧，则跳出循环
    if not ret:
        break
    # 使用YOLO模型对当前帧进行目标检测
    results = model.predict(frame)
    # 从检测结果中提取边界框坐标（xyxy格式），并转换为列表
    # 注意：这里原代码有重复赋值的错误，已修正
    boxes = results[0].boxes.xyxy.cpu().tolist()
    # 遍历所有检测到的边界框
    for box in boxes:
        # 从frame中截取边界框区域的图像
        # 格式为[y1:y2, x1:x2]，其中box[0]是x1, box[1]是y1, box[2]是x2, box[3]是y2
        obj = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
        # 对截取的图像区域进行高斯模糊处理，模糊核大小为70x70
        # 然后将模糊后的图像重新赋值给原位置，实现目标模糊效果
        frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])] = cv2.blur(obj, (70, 70))
    # 显示处理后的图像窗口
    cv2.imshow("YOLOv8 Inference", frame)
    # 等待键盘输入1毫秒，如果按下'q'键则退出循环
    if cv2.waitKey(1) == ord('q'):
        break
# 释放摄像头资源
cap.release()
# 关闭所有OpenCV创建的窗口
cv2.destroyAllWindows()
