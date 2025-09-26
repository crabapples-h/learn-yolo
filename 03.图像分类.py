from ultralytics import YOLO

# 加载预训练模型
model = YOLO('models/yolov8n-cls.pt')
model_names = model.names
results = model('./resources/图像分类.png', save=True)
for index in range(len(results)):
    item = results[index]
    probs = item.probs
    top_1 = probs.top1
    top_1_conf = probs.top1conf
    top_5 = probs.top5
    top_5_conf = probs.top5conf.numpy()
    top_5_names = [model_names[i] for i in top_5]
    print('第[{}]张图片分类结果:'.format(index))
    print('top1:[{}],可置信度:[{}],名称:[{}]'.format(top_1, top_1_conf, model.names[probs.top1]))
    print('top5:[{}],可置信度:[{}],名称:[{}]'.format(top_5, top_5_conf, top_5_names))
