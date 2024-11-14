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

# Import detection and classification modules
from ImageDetection.pest_detection import detect_pests
from ImageDetection.fruit_classfy import classify_fruit

app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# Configure upload file storage path
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize models and variables needed for camera detection
yolo_model = None
resnet_model = None
preprocess = None
fruits = ['apple', 'avocado', 'banana', 'cherry', 'kiwi', 'Mango', 'orange', 'pineapple', 'strawberry', 'watermelon']

def init_camera_detection_models():
    """
    Initialize YOLO and ResNet models for object detection and classification
    """
    global yolo_model, resnet_model, preprocess
    
    # Load YOLO model for object detection
    yolo_model = YOLO("yolov8n.pt")
    
    # Load ResNet model for fruit classification
    resnet_model = models.resnet50(weights=False)
    num_ftrs = resnet_model.fc.in_features
    resnet_model.fc = nn.Linear(num_ftrs, 10)
    
    # Load model weights
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_weights_path = os.path.join(current_dir, "CameraDetection/test_best_model_weights.pth")
    model_weights = torch.load(model_weights_path, map_location=torch.device('cpu'))
    resnet_model.load_state_dict(model_weights)
    resnet_model.eval()
    
    # Set up image preprocessing pipeline
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def process_frame(frame):
    """
    Process a single frame for object detection and classification
    
    Args:
        frame: Input image frame
    
    Returns:
        List of detection results containing bounding boxes, classes, and confidence scores
    """
    detection_results = []
    
    # Perform detection using YOLOv8
    results = yolo_model(frame, show=False)
    
    # Process detection results
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Crop and process detected region
            crop_img = frame[y1:y2, x1:x2]
            crop_img = Image.fromarray(crop_img)
            input_tensor = preprocess(crop_img).unsqueeze(0)
            
            # Perform classification using ResNet
            with torch.no_grad():
                output = resnet_model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
            
            detection_results.append({
                'bbox': [x1, y1, x2, y2],
                'class': fruits[pred],
                'confidence': float(box.conf[0])
            })
    
    return detection_results

def gen_frames():
    """
    Generator function for video streaming
    Yields processed frames with detection results drawn on them
    """
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        detection_results = process_frame(frame)
        
        # Draw detection results on frame
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

# API Routes
@app.route('/api/image-detection', methods=['POST'])
def image_detection():
    """
    API endpoint for image detection
    Handles uploaded images for pest detection and fruit classification
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename): # type: ignore
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            pest_result = detect_pests(filepath)
            fruit_result = classify_fruit(filepath)
            
            os.remove(filepath)
            
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
    """
    Health check endpoint to verify service status
    """
    return jsonify({'status': 'healthy'}), 200

# Camera detection routes
@app.route('/')
def index():
    """
    Render main page template
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    Endpoint for streaming video feed
    Returns a multipart response containing processed frames
    """
    return Response(gen_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/camera-detection', methods=['POST'])
def camera_detection():
    """
    Process frames from camera feed
    
    Expects:
        JSON with base64 encoded image in 'frame' field
    
    Returns:
        JSON with detection results
    """
    try:
        # Get base64 encoded image from request
        data = request.json
        image_data = data['frame'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process image and get detection results
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
    """
    Process a single image and return the results with annotations
    
    Expects:
        JSON with base64 encoded image in 'image' field
    
    Returns:
        JSON with base64 encoded processed image
    """
    try:
        # Get uploaded image data
        image_data = request.json['image']
        # Decode base64 image
        image_data = image_data.split(',')[1]
        image_bytes = base64.b64decode(image_data)
        
        # Convert to OpenCV format
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get detection results
        detection_results = process_frame(frame)
        
        # Draw detection results on frame
        for detection in detection_results:
            x1, y1, x2, y2 = detection['bbox']
            label = f"Fruit: {detection['class']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Convert back to base64 format
        _, buffer = cv2.imencode('.jpg', frame)
        processed_image = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({'processed_image': f'data:image/jpeg;base64,{processed_image}'})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize camera detection models before starting the server
    init_camera_detection_models()
    app.run(debug=True, port=5000)
