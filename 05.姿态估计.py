from ultralytics import YOLO

# 加载预训练模型
model = YOLO('models/yolo11n-pose.pt')
results = model('./resources/姿态估计',save= True)
for result in results:
    keypoints = result.keypoints
    print(keypoints)
