[English](README.md) | 简体中文

---
# 水果检测
本项目旨在实现水果识别和分类，并提供额外的病虫害检测功能。

## 项目概述
本项目提供了一个基于网络的实时水果检测和分类应用。它结合了现代网络技术和先进的人工智能模型，提供基于图像和摄像头的水果检测功能。

## 特性
### 核心功能
- 🔌丰富的核心功能
	- 支持上传图像检测和摄像头实时检测
	- 通过上传图像能够检测3种水果（苹果，香蕉，梨），并判断是否存在害虫
	- 通过摄像头能够实时检测10种水果，包括：
		- 苹果
		- 牛油果
		- 香蕉 
		- 樱桃 
		- 猕猴桃
		- 芒果 
		- 橘子 
		- 菠萝
		- 草莓
		- 西瓜
	- 返回每个检测到的对象的检测置信度

### 技术架构 
- 🌐 前端技术栈 
	- 基于React + TypeScript构建，确保代码类型安全 
	- 使用TailwindCSS实现响应式设计 
	- 实时摄像头画面捕捉和显示 
- 🛠️ 后端技术栈 
	- 基于Python + Flask构建RESTful API 
	- 支持异步任务处理 

### AI模型能力 
- 🤖 先进的目标检测与识别 
	- 采用YOLO v8实现实时目标检测 
	- 集成ResNet50进行特征提取和分类 

### 系统特性 
- ⚡ 性能优化 
	- 支持GPU加速 
	- 优化的模型推理速度

## 安装
1. 克隆仓库：
```bash
git clone https://github.com/Beichen-Hg/CV-project.git 
cd CV-project
```
2. 您需要将"frontend"和"backend"两个文件夹下载存放到同一个根目录下，否则前端代码无法正常执行！

### 环境设置
1. 前置组件：安装 JavaScript 环境 —— [Node.js](https://nodejs.org/)
2. 前端组件：
```bash
cd frontend
npm install react react-dom react-router-dom bootstrap
```
2. 后端组件：
```bash
cd backend
pip install flask flask-cors Pillow numpy opencv-python werkzeug torch torchvision ultralytics
```

## 使用方法
### 使用前端应用
1. 安装完所有的组件后，首先运行`backend/app.py`
2. 然后运行react-app，等待react-app弹窗，或手动输入[http://localhost:3000](http://localhost:3000)：
```bash
cd frontend
npm start
```

### 单独使用Detection功能
- 单独使用ImageDetection：确保安装了必要组件后运行`UI.py`文件
- 单独使用CameraDetection：确保安装了必要组件后运行`yolo检测+resnet50识别.py`。如果您想使用GPU加速摄像头检测（确保您的设备可以使用），可以将`# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`中的**hash tag** `#` 移除

### 补充说明
- 如果您想在前端网页上利用 GPU 加速摄像头检测（确保您的设备可以使用），可以在`app.py`文件中进行修改（将`'cpu'`改为`'cuda'`）：
```python
# 原代码
model_weights = torch.load(model_weights_path, map_location=torch.device('cpu'))

# 简单修改
model_weights = torch.load(model_weights_path, map_location=torch.device('cuda'))

# 更robust的修改方式（会检查是否有可用的GPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") model_weights = torch.load(model_weights_path, map_location=device)
```
- 如果您想训练自己的模型，可以尝试利用`train.py`文件，但是好几处代码也需要相应地修改，这里不做具体指导

## 配置说明
### 系统要求
- 操作系统：Windows 10/11、macOS 或 Linux
- 内存：最低8GB（建议16GB以获得更好性能）
- 显卡：支持CUDA的NVIDIA显卡（可选，用于GPU加速）
- 存储空间：至少2GB可用空间
- 摄像头：摄像头检测功能需要

### 环境要求
- Python 3.10+
- CUDA 11.7+ (如果使用GPU)
- Node.js 20+
- npm 10+

### 配置文件说明
项目配置文件位于 `backend/config.py`，包含以下配置项：
```python
# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')

# API配置
API_VERSION = 'v1'

# 允许的文件类型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# 图像处理配置
MAX_IMAGE_SIZE = (800, 800)  # 最大图像尺寸

# 响应状态码
HTTP_200_OK = 200
HTTP_400_BAD_REQUEST = 400
HTTP_500_INTERNAL_SERVER_ERROR = 500
```

### 模型配置
1. **YOLO模型**
   - 模型路径：`backend/CameraDetection/yolov8n.pt`
   - 置信度阈值：0.5
   - NMS阈值：0.45

2. **ResNet模型**
   - 预训练权重：`backend/CameraDetection/test_best_model_weights.pth`
   - 输入尺寸：224x224
   - 批处理大小：32

## 故障排除
### 常见问题
1. **摄像头不工作**
    - 检查浏览器是否启用了摄像头权限
    - 确保没有其他应用程序正在使用摄像头
2. **模型加载错误**
    - 验证所有模型文件是否在正确的位置
    - 检查CUDA是否正确安装（用于GPU使用）
3. **前端连接问题**
    - 确保前端和后端服务器都在运行
    - 检查端口（前端3000，后端5000）是否可用

## 联系方式
项目链接: [https://github.com/Beichen-Hg/CV-project](https://github.com/Beichen-Hg/CV-project)
