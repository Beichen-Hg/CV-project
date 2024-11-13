English | [简体中文](README_ch.md)

---
# Fruit Detection
The goal of this project is to implement fruit recognition and classification, and provide additional pest detection functionality.

## Project Overview 
This project provides a web-based application for real-time fruit detection and classification. It combines modern web technologies with advanced AI models to deliver both image-based and camera-based fruit detection capabilities.

## Features
### Core Functionality
- 🔌 Rich core features
	- Supports both image upload detection and real-time camera detection
	- Can detect 3 types of fruits(Apple, Banana, Pear) and determine the presence of pests through image upload
	- Can detect 10 types of fruits in real-time through camera, including:
		- Apple
		- Avocado
		- Banana 
		- Cherry 
		- Kiwi 
		- Mango 
		- Orange 
		- Pineapple 
		- Strawberry
		- Watermelon
	- Returns detection confidence scores for each detected object

### Technical Architecture 
- 🌐 Frontend Tech Stack 
	- Built with React + JavaScript ensuring code type safety 
	- Responsive design implemented using BootstrapCSS 
	- Real-time camera feed capture and display 
- 🛠️ Backend Tech Stack 
	- RESTful API built with Python + Flask 
	- Supports asynchronous task processing 

### AI Model Capabilities 
- 🤖 Advanced Object Detection and Recognition 
	- Real-time object detection using YOLO v8 
	- Integrated ResNet50 for feature extraction and classification 

### System Features 
- ⚡ Performance Optimization 
	- Supports GPU acceleration 
	- Optimized model inference speed

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Beichen-Hg/CV-project.git 
cd CV-project
```
2. You need to download both the `frontend` and `backend` folders and **place them in the same root directory**, otherwise the frontend code will not execute properly!
3. You **must** additionally download the files from the `master` branch repositorie and place the `CameraDetection` folder under the `backend` folder.
4. Furthermore, you **must** create a folder named `uploads` in both the project root directory and the `backend` directory to temporarily store uploaded images!

The first three levels of the directory structure should look like the following (the "node_modules" folder will be generated in the environment setup section):
```text
project-root/
├── frontend/
│   ├── node_modules/
│   ├── public/
│   ├── src/
│   ├── .gitignore
│   ├── package.json
│   └── package-lock.json
├── backend/
│   ├── CameraDetection/
│   ├── ImageDetection/
│   ├── uploads/
│   ├── app.py
│   ├── config.py
│   └── utils.py
├── uploads/
├── README.md
├── README_ch.md
└── yolov8n.pt
```

### Environment Setup
1. Prerequisites: Install JavaScript environment — [Node.js](https://nodejs.org/)
2. Frontend components:
```bash
cd frontend
npm install react react-dom react-router-dom bootstrap
```
2. Backend components:
```bash
cd backend
pip install flask flask-cors Pillow numpy opencv-python werkzeug torch torchvision ultralytics
```

## Usage
### Using the Frontend Application
1. After installing all components, first run `backend/app.py`
2. Then run the react-app, wait for the react-app popup, or manually enter [http://localhost:3000](http://localhost:3000):
```bash
cd frontend
npm start
```

### Using Detection Features Separately
- To use ImageDetection separately: Ensure necessary components are installed and run the `UI.py` file
- To use CameraDetection separately: Ensure necessary components are installed and run `yolo检测+resnet50识别.py`. If you want to use GPU acceleration for camera detection (ensure your device supports it), you can remove the **hash tag** `#` from `# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`

### Additional Notes
- If you want to use GPU acceleration for camera detection on the frontend webpage (ensure your device supports it), you can modify the `app.py` file (change `'cpu'` to `'cuda'`):
```python
# Original code
model_weights = torch.load(model_weights_path, map_location=torch.device('cpu'))

# Simple modification
model_weights = torch.load(model_weights_path, map_location=torch.device('cuda'))

# More robust modification (checks for available GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
model_weights = torch.load(model_weights_path, map_location=device)
```
- If you want to train your own model, you can try using the `train.py` file, but several parts of the code need to be modified accordingly. Specific guidance is not provided here.

## Configuration Details
### System Requirements
- Operating System: Windows 10/11, macOS, or Linux
- RAM: Minimum 8GB (16GB recommended for better performance)
- GPU: NVIDIA GPU with CUDA support (optional, for GPU acceleration)
- Storage: At least 2GB free space
- Webcam: Required for camera detection feature

### Environment Requirements
- Python 3.10+
- CUDA 11.7+ (if using GPU)
- Node.js 20+
- npm 10+

### Configuration File Description
The project configuration file is located at `backend/config.py`, containing the following configuration items:
```python
# Base path configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# API configuration
API_VERSION = 'v1'

# Allowed file types
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Image processing configuration
MAX_IMAGE_SIZE = (800, 800)  # Maximum image size

# Response status codes
HTTP_200_OK = 200
HTTP_400_BAD_REQUEST = 400
HTTP_500_INTERNAL_SERVER_ERROR = 500
```

### Model Configuration
1. **YOLO Model**
   - Model path: `backend/CameraDetection/yolov8n.pt`
   - Confidence threshold: 0.5
   - NMS threshold: 0.45

2. **ResNet Model**
   - Pre-trained weights: `backend/CameraDetection/test_best_model_weights.pth`
   - Input size: 224x224
   - Batch size: 32

## Troubleshooting
### Common Issues
1. **Camera not working**
   - Check if your browser has camera permissions enabled
   - Ensure no other application is using the camera
   
2. **Model loading errors**
   - Verify all model files are in the correct locations
   - Check if CUDA is properly installed (for GPU usage)

3. **Frontend connection issues**
   - Ensure both frontend and backend servers are running
   - Check if the ports (3000 for frontend, 5000 for backend) are available

## Contact Information
Project link: [https://github.com/Beichen-Hg/CV-project](https://github.com/Beichen-Hg/CV-project)
