from ultralytics import YOLO
import os

os.environ['ULTRALYTICS_PLOTS'] = 'False'  # 禁用绘图环境变量


def safe_training():
    """安全训练，彻底避免NumPy问题"""

    model = YOLO('yolov8n-pose.pt')

    results = model.train(
        data='./dataset1/train.yaml',
        epochs=100,
        imgsz=320,
        batch=2,
        device='mps',
        workers=0,  # 设置为0避免多进程问题

        # 关键：彻底禁用所有可能触发NumPy bug的功能
        plots=False,  # 禁用绘图
        save_json=False,  # 禁用JSON保存
        verbose=True,

        # 简化所有参数
        lr0=0.001,
        pose=2.0,
        kobj=1.5,

        # 关闭数据增强
        augment=False,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.0,
        degrees=0.0,
        translate=0.0,
        scale=0.0,
        fliplr=0.0,
    )

    return results


print("开始安全训练...")
safe_training()