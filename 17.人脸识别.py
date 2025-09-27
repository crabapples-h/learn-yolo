import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
from numpy.linalg import norm

# 初始化模型
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))

# 创建一个数据库来存储已知人脸的特征和姓名
face_database = {}

# --- 注册阶段：将已知人脸加入数据库 ---
def register_face(image_path, person_name):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) != 1:
        print(f"警告：在 {image_path} 中未检测到或检测到多张人脸，跳过。")
        return
    embedding = faces[0].embedding
    face_database[person_name] = embedding
    print(f"成功注册：{person_name}")

# 注册多个人
img1 = cv2.imread('./resources/face/01.png')

register_face('./resources/face/01.png', '01')
register_face('./resources/face/02.png', '02')

# --- 识别阶段：识别未知图片中的人 ---
def recognize_face(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if len(faces) == 0:
        print("未检测到人脸。")
        return

    for i, face in enumerate(faces):
        unknown_embedding = face.embedding
        max_sim = -1
        identity = "未知"

        # 与数据库中的每个人脸进行比对
        for name, known_embedding in face_database.items():
            cos_sim = unknown_embedding @ known_embedding.T / (norm(unknown_embedding) * norm(known_embedding))
            if cos_sim > max_sim:
                max_sim = cos_sim
                identity = name

        # 根据阈值决定最终身份
        threshold = 0.6
        if max_sim < threshold:
            identity = "未知"

        print(f"人脸 {i+1}: 识别为 【{identity}】, 相似度：{max_sim:.4f}")

# 识别一张新图片
recognize_face('./resources/face/16.jpg')