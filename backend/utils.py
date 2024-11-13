import os
from PIL import Image
import numpy as np
from config import ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """
    处理上传的图像文件
    """
    try:
        # 打开图像
        with Image.open(image_path) as img:
            # 转换为RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 调整图像大小
            img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
            
            # 转换为numpy数组
            img_array = np.array(img)
            
            return img_array
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def cleanup_file(filepath):
    """
    清理临时文件
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up file {filepath}: {str(e)}")