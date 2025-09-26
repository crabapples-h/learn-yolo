import json
import os


def labelme_to_yolo(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    # 自动提取所有类别并建立索引
    classes = list(set(shape['label'] for shape in data['shapes']))
    classes.sort()  # 按字母排序保持一致性

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    yolo_lines = []
    for shape in data['shapes']:
        class_name = shape['label']
        class_id = classes.index(class_name)

        points = shape['points']
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        # 归一化坐标
        x_center = (x_min + x_max) / 2 / img_width
        y_center = (y_min + y_max) / 2 / img_height
        width = (x_max - x_min) / img_width
        height = (y_max - y_min) / img_height

        yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 保存标签文件
    txt_filename = os.path.splitext(os.path.basename(json_file))[0] + '.txt'
    txt_path = os.path.join(output_dir, txt_filename)

    with open(txt_path, 'w') as f:
        f.write('\n'.join(yolo_lines))

    # 保存类别文件
    with open(os.path.join(output_dir, 'classes.txt'), 'w') as f:
        f.write('\n'.join(classes))

    return classes


# 使用
json_file = "./dataset1/json/1.json"
output_dir = "./dataset1/labels"
classes = labelme_to_yolo(json_file, output_dir)
print("检测到的类别:", classes)
