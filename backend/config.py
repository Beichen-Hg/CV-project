import os

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