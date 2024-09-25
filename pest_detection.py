import cv2
import numpy as np

def detect_pests(image_path):

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or path is incorrect.")
        return

    # 将HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # pest color
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # mask
    mask = cv2.inRange(hsv_image, lower_black, upper_black)

    # 
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # white ratio
    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    white_ratio = (white_pixels / total_pixels) * 100

    # pest exist or not
    if white_ratio > 0.1: 
        result = "Pests detected!"
    else:
        result = "No pests detected."

    # show result
    #cv2.imshow("Original", image)
    #cv2.imshow("Pests Detected", cv2.bitwise_and(image, image, mask=mask))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return result

# test function
#result = detect_pests("apple1.png")
#print(result)
