import cv2
import numpy as np

def classify_fruit(image_path):
    
    image = cv2.imread(image_path)
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # color range
    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([25, 100, 100])
    yellow_upper = np.array([35, 255, 255])
    green_lower = np.array([30, 100, 100])
    green_upper = np.array([50, 255, 255])

    # color mask
    red_mask = cv2.inRange(hsv_image, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv_image, green_lower, green_upper)

    # pixel ratio
    red_ratio = np.sum(red_mask) / (image.shape[0] * image.shape[1])
    yellow_ratio = np.sum(yellow_mask) / (image.shape[0] * image.shape[1])
    green_ratio = np.sum(green_mask) / (image.shape[0] * image.shape[1])

    # classfy
    if max(red_ratio, yellow_ratio, green_ratio) == red_ratio:
        return "Apple"
    elif max(red_ratio, yellow_ratio, green_ratio) == yellow_ratio:
        return "Banana"
    elif max(red_ratio, yellow_ratio, green_ratio) == green_ratio:
        return "Pear"

# test function
#result = classify_fruit("pear2.png")
#print("The fruit is a:", result)
