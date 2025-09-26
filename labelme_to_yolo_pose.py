import json
import os


def labelme_to_yolo_pose(json_file, output_dir):
    with open(json_file, 'r') as f:
        data = json.load(f)

    img_width = data['imageWidth']
    img_height = data['imageHeight']

    # 提取手部边界框和关键点
    bbox = None
    keypoints = {}

    for shape in data['shapes']:
        if shape['shape_type'] == 'rectangle' and shape['label'] == 'hand':
            points = shape['points']
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            x_min, x_max = min(x_coords), max(x_coords)
            y_min, y_max = min(y_coords), max(y_coords)

            x_center = (x_min + x_max) / 2 / img_width
            y_center = (y_min + y_max) / 2 / img_height
            width = (x_max - x_min) / img_width
            height = (y_max - y_min) / img_height
            bbox = (x_center, y_center, width, height)

        elif shape['shape_type'] == 'point':
            label = shape['label']
            x, y = shape['points'][0]
            keypoints[label] = (x / img_width, y / img_height)

    # 关键点顺序（按JSON中的标签）
    keypoint_order = ['f_1_1', 'f_1_2', 'f_2_1', 'f_2_2', 'f_3_2', 'f_3_1',
                      'f_4_1', 'f_4_2', 'f_5_2', 'f_5_1', 'f']

    if bbox:
        yolo_line = f"0 {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}"

        for kp_name in keypoint_order:
            if kp_name in keypoints:
                kx, ky = keypoints[kp_name]
                yolo_line += f" {kx:.6f} {ky:.6f} 2"
            else:
                yolo_line += " 0 0 0"

        # 保存为txt文件
        txt_filename = os.path.splitext(os.path.basename(json_file))[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)

        with open(txt_path, 'w') as f:
            f.write(yolo_line)

        print(f"转换完成: {txt_filename}")


# 使用
json_file = "./dataset1/json/1.json"
output_dir = "./dataset1/labels"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

labelme_to_yolo_pose(json_file, output_dir)


