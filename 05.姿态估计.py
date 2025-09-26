from ultralytics import YOLO

# 加载预训练模型
model = YOLO('models/yolo11n-pose.pt')
results = model('./resources/姿态估计',save= True)
# 1.鼻子 2.左眼 3.右眼 4.左耳 5.右耳 6.左肩 7.右肩 8.左手肘 9.右肘 10.左手腕
# 11.右腕 12.左髋 13.右髋 14.左膝 15.右膝 16. 左脚踝 17.右脚踝
for result in results:
    keypoints = result.keypoints
    print(keypoints)
