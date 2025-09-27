import insightface
from insightface.app import FaceAnalysis
import cv2

# 1. 创建人脸分析应用
# 这会自动下载预训练模型（第一次运行时会下载，约 200MB）
app = FaceAnalysis(name='buffalo_l') # 'buffalo_l' 是当前最准的模型集合
app.prepare(ctx_id=0, det_size=(640, 640)) # ctx_id=0 表示使用CPU， ctx_id=1/gpu_id 表示使用GPU

# 2. 加载图片
img1 = cv2.imread('./resources/face/01.png')
# img2 = cv2.imread('./resources/face/02.png')
img2 = cv2.imread('./resources/face/16.jpg')

# 3. 进行人脸检测和特征提取
faces1 = app.get(img1)
faces2 = app.get(img2)

# 检查是否检测到人脸
if len(faces1) == 0 or len(faces2) == 0:
    print("未在图片中检测到人脸！")
    exit()

# 取每张图片中检测到的第一个人脸
face1 = faces1[0]
face2 = faces2[0]

# 4. 获取人脸特征向量（嵌入）
embedding1 = face1.embedding
embedding2 = face2.embedding

# 5. 计算特征向量之间的相似度（使用余弦相似度）
from numpy.linalg import norm
cos_sim = embedding1 @ embedding2.T / (norm(embedding1) * norm(embedding2))
print(f"人脸相似度（余弦）： {cos_sim:.4f}")

# 6. 根据阈值判断是否为同一个人
threshold = 0.6  # 常用阈值，可根据场景调整（范围一般在0.2-0.8之间，值越大判断越严格）
if cos_sim > threshold:
    print("判断为同一个人！")
else:
    print("判断为不同人。")