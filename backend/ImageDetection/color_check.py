import cv2
import numpy as np

def get_hsv_values(image_path):
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or path is incorrect.")
        return None

    # Convert image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Select ROI (Region of Interest)
    roi = cv2.selectROI("Select ROI", hsv_image, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("Error: No region selected.")
        cv2.destroyAllWindows()
        return None

    roi_cropped = hsv_image[y:y+h, x:x+w]

    # Calculate average HSV values for the selected region
    if roi_cropped.size == 0:
        print("Error: Selected region is empty.")
        return None
    else:
        average_hsv = np.mean(roi_cropped, axis=(0, 1))
        # Output average HSV values
        print("Average HSV values for the selected region:")
        print("H: {:.2f}, S: {:.2f}, V: {:.2f}".format(average_hsv[0], average_hsv[1], average_hsv[2]))
        return average_hsv

# Usage example
image_path = r"C:\Users\Bertram\Desktop\project cv\backend\ImageDetection\pear2.png"
result = get_hsv_values(image_path)
if result:
    print(f"HSV values: {result}")

