from flask import Flask, request, jsonify, render_template, Response
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import io
import base64
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn as nn
from ultralytics import YOLO

# 导入检测和分类模块
from ImageDetection.pest_detection import detect_pests
from ImageDetection.fruit_classfy import classify_fruit

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 配置上传文件的存储路径
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 初始化摄像头检测所需的模型和变量
yolo_model = None
resnet_model = None
preprocess = None
fruits = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'Mango', 'orange', 'pineapple', 'strawberry', 'watermelon']

def init_camera_detection_models():
    global yolo_model, resnet_model, preprocess
    
    # 加载YOLO模型
    yolo_model = YOLO("yolov8n.pt")
    
    # 加载ResNet模型
    resnet_model = models.resnet50(weights=False)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 10)
    
    # 加载模型权重
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_weights_path = os.path.join(current_dir, "CameraDetection/test_best_model_weights.pth")
    model_weights = torch.load(model_weights_path, map_location=torch.device('cpu'))
    resnet_model.load_state_dict(model_weights)
    resnet_model.eval()
    
    # 定义预处理流程
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_frame(frame):
    detection_results = []
    
    # 使用YOLOv8进行检测
    results = yolo_model(frame, show=False)
    
    # 处理检测结果
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 裁剪并处理检测到的区域
            crop_img = frame[y1:y2, x1:x2]
            crop_img = Image.fromarray(crop_img)
            input_tensor = preprocess(crop_img).unsqueeze(0)
            
            # 使用ResNet进行分类
            with torch.no_grad():
                output = resnet_model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
            
            # 添加检测结果
            detection_results.append({
                'bbox': [x1, y1, x2, y2],
                'class': fruits[pred],
                'confidence': float(box.conf[0])
            })
    
    return detection_results

def gen_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # 获取检测结果
            detection_results = process_frame(frame)
            
            # 在图像上绘制检测结果
            for detection in detection_results:
                x1, y1, x2, y2 = detection['bbox']
                label = f"Fruit: {detection['class']}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# ImageDetection的API路由
@app.route('/api/image-detection', methods=['POST'])
def image_detection():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            # 保存文件
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # 进行害虫检测
            pest_result = detect_pests(filepath)
            
            # 进行水果分类
            fruit_result = classify_fruit(filepath)
            
            # 删除临时文件
            os.remove(filepath)
            
            # 返回结果
            return jsonify({
                'success': True,
                'pest_detection': pest_result,
                'fruit_classification': fruit_result
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'An error occurred during processing'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'}), 200

# 摄像头检测路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera-detection', methods=['POST'])
def camera_detection():
    try:
        # 从请求中获取base64编码的图像
        data = request.json
        image_data = data['frame'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # 转换为OpenCV格式
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 处理图片并获取检测结果
        detection_results = process_frame(frame)
        
        return jsonify({
            'success': True,
            'detections': detection_results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/process_image', methods=['POST'])
def process_image():
    try:
        # 获取上传的图片数据
        image_data = request.json['image']
        # 解码base64图片
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # 转换为OpenCV格式
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 获取检测结果
        detection_results = process_frame(frame)
        
        # 在图像上绘制检测结果
        for detection in detection_results:
            x1, y1, x2, y2 = detection['bbox']
            label = f"Fruit: {detection['class']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 转换回base64格式
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'processed_image': f'data:image/jpeg;base64,{processed_image}'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    init_camera_detection_models()  # 初始化摄像头检测模型
    app.run(debug=True, port=5000)