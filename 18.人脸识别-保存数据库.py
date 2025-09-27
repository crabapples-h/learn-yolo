import insightface
from insightface.app import FaceAnalysis
import cv2
import numpy as np
import sqlite3
import os
from numpy.linalg import norm

# 初始化InsightFace模型
app = FaceAnalysis(name='buffalo_l')
app.prepare(ctx_id=0, det_size=(640, 640))


class FaceDatabase:
    def __init__(self, db_path='face_database.db'):
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                embedding BLOB NOT NULL,
                created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        conn.close()
        print(f"数据库初始化完成: {self.db_path}")

    def add_face(self, name, embedding):
        """添加人脸到数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 将numpy数组转换为bytes
        embedding_blob = embedding.tobytes()

        try:
            cursor.execute(
                'INSERT INTO faces (name, embedding) VALUES (?, ?)',
                (name, embedding_blob)
            )
            conn.commit()
            print(f"✅ 成功注册: {name}")
            return True
        except sqlite3.IntegrityError:
            print(f"⚠️ 姓名已存在: {name}")
            return False
        finally:
            conn.close()

    def get_all_faces(self):
        """获取所有人脸数据"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT name, embedding FROM faces')
        results = cursor.fetchall()
        conn.close()

        face_database = {}
        for name, embedding_blob in results:
            # 将bytes转换回numpy数组
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            face_database[name] = embedding

        print(f"📊 从数据库加载了 {len(face_database)} 个人脸特征")
        return face_database

    def delete_face(self, name):
        """删除人脸"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('DELETE FROM faces WHERE name = ?', (name,))
        affected_rows = cursor.rowcount
        conn.commit()
        conn.close()

        if affected_rows > 0:
            print(f"🗑️ 已删除: {name}")
            return True
        else:
            print(f"❌ 未找到: {name}")
            return False

    def face_exists(self, name):
        """检查人脸是否存在"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT 1 FROM faces WHERE name = ?', (name,))
        exists = cursor.fetchone() is not None
        conn.close()

        return exists


# 创建全局数据库实例
face_db = FaceDatabase()


def register_face(image_path, person_name):
    """注册人脸"""
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return False

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return False

    # 使用InsightFace检测人脸
    faces = app.get(img)

    if len(faces) == 0:
        print(f"❌ 在 {image_path} 中未检测到人脸")
        return False
    elif len(faces) > 1:
        print(f"⚠️ 在 {image_path} 中检测到 {len(faces)} 张人脸，使用第一张")

    # 提取人脸特征
    embedding = faces[0].embedding
    print(f"📐 提取到特征向量，维度: {embedding.shape}")

    # 保存到数据库
    return face_db.add_face(person_name, embedding)


def register_faces_from_folder(folder_path):
    """从文件夹批量注册人脸"""
    if not os.path.exists(folder_path):
        print(f"❌ 文件夹不存在: {folder_path}")
        return

    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    registered_count = 0

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(supported_formats):
            # 使用文件名（不含扩展名）作为人名
            person_name = os.path.splitext(filename)[0]
            image_path = os.path.join(folder_path, filename)

            if register_face(image_path, person_name):
                registered_count += 1

    print(f"🎉 批量注册完成，共注册 {registered_count} 个人脸")


def recognize_face(image_path, threshold=0.6):
    """识别人脸"""
    # 检查图片是否存在
    if not os.path.exists(image_path):
        print(f"❌ 图片不存在: {image_path}")
        return

    # 读取图片
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ 无法读取图片: {image_path}")
        return

    # 从数据库加载所有人脸特征
    face_database = face_db.get_all_faces()

    if not face_database:
        print("❌ 数据库中无人脸数据，请先注册人脸")
        return

    # 使用InsightFace检测人脸
    faces = app.get(img)

    if len(faces) == 0:
        print("❌ 未检测到人脸")
        return

    print(f"🔍 检测到 {len(faces)} 张人脸，开始识别...")

    for i, face in enumerate(faces):
        unknown_embedding = face.embedding
        max_sim = -1
        identity = "未知"
        best_match_name = None

        # 与数据库中的每个人脸进行比对
        for name, known_embedding in face_database.items():
            # 计算余弦相似度
            cos_sim = unknown_embedding @ known_embedding.T / (norm(unknown_embedding) * norm(known_embedding))

            if cos_sim > max_sim:
                max_sim = cos_sim
                identity = name
                best_match_name = name

        # 根据阈值决定最终身份
        if max_sim < threshold:
            identity = "未知"

        # 输出结果
        status = "✅" if identity != "未知" else "❓"
        print(f"{status} 人脸 {i + 1}: 识别为 【{identity}】, 相似度: {max_sim:.4f}")

        # 显示匹配详情
        if identity != "未知":
            matched_embedding = face_database[best_match_name]
            print(f"   匹配特征: {best_match_name}, 范数: {norm(matched_embedding):.4f}")


def list_registered_faces():
    """列出所有已注册的人脸"""
    face_database = face_db.get_all_faces()

    if not face_database:
        print("📭 数据库为空")
        return

    print(f"\n📋 已注册的人脸 ({len(face_database)} 个):")
    for i, name in enumerate(face_database.keys(), 1):
        print(f"  {i}. {name}")


def delete_face(person_name):
    """删除指定人脸"""
    return face_db.delete_face(person_name)


# 使用示例
if __name__ == "__main__":
    print("=" * 50)
    print("🎭 InsightFace 人脸识别系统 (SQLite版本)")
    print("=" * 50)

    # 1. 批量注册人脸
    print("\n1. 📁 批量注册人脸")
    register_faces_from_folder('./resources/face')

    # 2. 单个注册
    print("\n2. 👤 单个注册")
    register_face('./resources/face/01.png', '张三')
    register_face('./resources/face/02.png', '李四')

    # 3. 查看已注册的人脸
    print("\n3. 📊 已注册人脸列表")
    list_registered_faces()

    # 4. 识别人脸
    print("\n4. 🔍 人脸识别测试")
    recognize_face('./resources/face/16.jpg', threshold=0.6)

    # 5. 删除测试
    # print("\n5. 🗑️ 删除人脸测试")
    # delete_face('测试删除')

    print("\n" + "=" * 50)
    print("✨ 程序执行完成")
    print("=" * 50)