import cv2
from ultralytics import YOLO
import numpy as np

# 加载预训练模型
model = YOLO('models/yolov8n-seg.pt')
print(model.names)
img_path = "./resources/图像分类.png"
results = model(img_path, save=True)
image = cv2.imread(img_path)
# 创建掩码,如需单独分割,则在循环中创建多个mask
mask = np.zeros(image.shape[:2], dtype=np.uint8)
for result in results:
    masks = result.masks
    points_array = masks.xy
    for points in points_array:
        points = points.astype(int)
        cv2.fillPoly(mask, [points], 255)
        green_bg = np.full_like(image, (0, 255, 0))
        result = np.where(mask[:, :, np.newaxis] == 255, image, green_bg)
        cv2.imshow('抠图', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
