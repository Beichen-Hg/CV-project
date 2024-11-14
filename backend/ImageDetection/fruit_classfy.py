import cv2
import numpy as np

def classify_fruit(image_path):
    # Read the input image
    image = cv2.imread(image_path)
    
    # Convert BGR image to HSV color space for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define color ranges in HSV format for different fruits
    # Format: [Hue, Saturation, Value]
    red_lower = np.array([0, 120, 70])     # Lower bound for red color
    red_upper = np.array([10, 255, 255])   # Upper bound for red color
    yellow_lower = np.array([25, 100, 100]) # Lower bound for yellow color
    yellow_upper = np.array([35, 255, 255]) # Upper bound for yellow color
    green_lower = np.array([30, 100, 100])  # Lower bound for green color
    green_upper = np.array([50, 255, 255])  # Upper bound for green color

    # Create binary masks for each color range
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # Calculate the ratio of each color in the image
    # Divide the sum of white pixels in mask by total number of pixels
    red_ratio = np.sum(red_mask) / (image.shape[0] * image.shape[1])
    yellow_ratio = np.sum(yellow_mask) / (image.shape[0] * image.shape[1])
    green_ratio = np.sum(green_mask) / (image.shape[0] * image.shape[1])

    # Classify the fruit based on the dominant color
    if max(red_ratio, yellow_ratio, green_ratio) == red_ratio:
        return "Apple"     # Red dominant -> Apple
    elif max(red_ratio, yellow_ratio, green_ratio) == yellow_ratio:
        return "Banana"    # Yellow dominant -> Banana
    elif max(red_ratio, yellow_ratio, green_ratio) == green_ratio:
        return "Pear"      # Green dominant -> Pear

