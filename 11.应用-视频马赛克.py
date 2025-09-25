import cv2
from ultralytics import YOLO
from moviepy import VideoFileClip  # 用于处理音频（仅视频文件需要）

def apply_gaussian_blur(region, blur_strength=25):
    """对目标区域应用高斯模糊"""
    return cv2.GaussianBlur(region, (blur_strength, blur_strength), 0)

def process_video(input_video, output_video, model, blur_strength=25, target_classes=None):
    """处理视频并应用高斯模糊"""
    # 提取原始视频的音频（仅用于视频文件）
    original_clip = VideoFileClip(input_video)
    audio = original_clip.audio

    # 读取视频
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建临时无音频视频
    temp_output = "temp_blurred_no_audio.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 检测
        results = model(frame)

        # 遍历检测结果
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # 检测框坐标
            cls_id = int(box.cls.item())  # 类别ID

            # 如果指定了目标类别，只模糊这些类别
            if target_classes is None or cls_id in target_classes:
                # 提取检测区域并模糊
                roi = frame[y1:y2, x1:x2]
                blurred_roi = apply_gaussian_blur(roi, blur_strength)
                frame[y1:y2, x1:x2] = blurred_roi

        # 写入输出视频
        out.write(frame)

        # 显示实时结果（可选）
        cv2.imshow("YOLO Blur Detection", frame)
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # 合并音频（仅视频文件需要）
    blurred_clip = VideoFileClip(temp_output)
    final_clip = blurred_clip.with_audio(audio)
    final_clip.write_videofile(output_video, codec="libx264", audio_codec="aac")

    # 删除临时文件（可选）
    import os
    os.remove(temp_output)

def main():
    # 加载 YOLO 模型（可以是 yolov8n.pt, yolov8s.pt 等）
    model = YOLO("runs/detect/train4/weights/best.pt")  # 替换成你的模型

    # 输入视频文件（如果是摄像头，设为 0）
    input_video = "resources/input1.mp4"  # 或 0（摄像头）
    output_video = "output_blurred.mp4"

    # 设置高斯模糊强度（越大越模糊）
    blur_strength = 79

    # 指定要模糊的类别（可选，None 表示模糊所有检测到的目标）
    # COCO 数据集类别：0: person, 1: bicycle, 2: car, ..., 参见 https://cocodataset.org
    target_classes = [0]  # 示例：只模糊人（person）

    if input_video == 0:
        # 摄像头实时检测（无音频）
        cap = cv2.VideoCapture(0)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 检测 + 模糊
            results = model(frame)
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls.item())
                if target_classes is None or cls_id in target_classes:
                    roi = frame[y1:y2, x1:x2]
                    frame[y1:y2, x1:x2] = apply_gaussian_blur(roi, blur_strength)

            cv2.imshow("Live Blur Detection", frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
    else:
        # 处理视频文件（保留音频）
        process_video(input_video, output_video, model, blur_strength, target_classes)

if __name__ == "__main__":
    main()