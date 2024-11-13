import cv2
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
from ultralytics import YOLO
import os

# 切换到GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载训练好的YOLO模型
yolo_model = YOLO("yolov8n.pt")

# 加载训练好的ResNet模型
resnet_model = models.resnet50(weights=False)   # pretrained=False表示不加载预训练的权重

# 如果你修改了模型的某些层（例如最后的全连接层），你需要在这里重新定义它们
num_ftrs = resnet_model.fc.in_features
resnet_model.fc = nn.Linear(num_ftrs, 10)  # num_classes是你的数据集的类别数

# 获取当前脚本的目录
current_dir = os.path.dirname(os.path.abspath(__file__))

# 构建模型权重文件的完整路径
model_weights_path = os.path.join(current_dir, "test_best_model_weights.pth")

# 把重新训练的输出层参数导入resnet模型
model_weights = torch.load(model_weights_path)

# 将权重应用到模型
resnet_model.load_state_dict(model_weights)

# 确保在推理模式下使用模型
resnet_model.eval()

# 定义图像预处理流程
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),  # 例如，将图片大小调整为224*224
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 使用YOLOv8进行检测
    results = yolo_model(frame, show = False)
    # 绘制检测框和类别标签
    for result in results:
        boxes = result.boxes
        for i, box in enumerate(boxes):
            # 获取边界框坐标
            x1, y1, x2, y2 = box.xyxy[0]
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # 裁剪边框区域
            crop_img = frame[int(y1):int(y2), int(x1):int(x2)]
            # 将裁剪的图像转换为PIL图像
            crop_img = Image.fromarray(crop_img)
            # 图像预处理
            input_tensor = preprocess(crop_img).unsqueeze(0)  # 添加batch维

            # 使用ResNet模型进行分类
            with torch.no_grad():
                output = resnet_model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
            # 在边框上显示分类结果
            fruits = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'Mango', 'orange', 'pineapple', 'strawberry',
                      'watermelon']
            cv2.putText(frame, f'Fruit: {fruits[pred]}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('frame', frame)

    # 按 'q' 键退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()