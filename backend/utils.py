import os
from PIL import Image
import numpy as np
from config import ALLOWED_EXTENSIONS, MAX_IMAGE_SIZE

def allowed_file(filename):
    """
    Check if the file extension is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path):
    """
    Process the uploaded image file
    """
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Convert to RGB mode
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Resize image
            img.thumbnail(MAX_IMAGE_SIZE, Image.LANCZOS)
            
            # Convert to numpy array
            img_array = np.array(img)
            
            return img_array
    except Exception as e:
        raise Exception(f"Error processing image: {str(e)}")

def cleanup_file(filepath):
    """
    Clean up temporary files
    """
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
    except Exception as e:
        print(f"Error cleaning up file {filepath}: {str(e)}")
