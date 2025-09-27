def verify_annotation_quality():
    """验证关键点标注质量"""

    import cv2
    import numpy as np
    from pathlib import Path
    import yaml

    # 加载配置
    with open('./dataset1/train.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # 加载一张图像和标注
    images_dir = Path('./dataset1/train/images')
    labels_dir = Path('./dataset1/train/labels')

    image_files = list(images_dir.glob('*.jpg'))[:3]  # 检查前3张

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"

        if not label_path.exists():
            continue

        # 读取图像
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]

        # 读取标注
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) != 68:  # 5 + 21*3
                continue

            # 解析边界框
            x_center, y_center, width, height = map(float, parts[1:5])
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)

            # 绘制边界框
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 绘制关键点
            keypoints = list(map(float, parts[5:]))
            for i in range(0, len(keypoints), 3):
                x, y, vis = keypoints[i], keypoints[i + 1], int(keypoints[i + 2])
                if vis > 0:  # 只绘制可见点
                    px = int(x * w)
                    py = int(y * h)
                    cv2.circle(img, (px, py), 3, (0, 0, 255), -1)
                    cv2.putText(img, str(i // 3), (px, py - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        # 保存可视化结果
        output_path = f"annotation_check_{img_path.stem}.jpg"
        cv2.imwrite(output_path, img)
        print(f"保存标注检查: {output_path}")


print("检查标注质量...")
verify_annotation_quality()