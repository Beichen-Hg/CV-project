import cv2
import numpy as np

def detect_pests(image_path):
    """
    Detect pests in an image by identifying dark-colored regions.
    
    Args:
        image_path (str): Path to the input image file
        
    Returns:
        str: Detection result message
    """
    # Read the input image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or path is incorrect.")
        return

    # Convert the image from BGR to HSV color space for better color detection
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the color range for pest detection (dark/black colors)
    # HSV format: [Hue, Saturation, Value]
    lower_black = np.array([0, 0, 0])      # Lower bound for dark colors
    upper_black = np.array([180, 255, 50]) # Upper bound for dark colors

    # Create a binary mask for the specified color range
    mask = cv2.inRange(hsv_image, lower_black, upper_black)

    # Apply morphological operations to reduce noise and improve detection
    kernel = np.ones((5, 5), np.uint8)
    # Close operation (dilation followed by erosion) to fill small holes
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Open operation (erosion followed by dilation) to remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Calculate the percentage of white pixels in the mask
    # White pixels represent potential pest areas
    white_pixels = np.sum(mask == 255)
    total_pixels = mask.size
    white_ratio = (white_pixels / total_pixels) * 100

    # Determine if pests are present based on the white pixel ratio threshold
    if white_ratio > 0.1:  # Threshold can be adjusted based on requirements
        result = "Pests detected!"
    else:
        result = "No pests detected."

    # Visualization code (commented out)
    # Uncomment to display the original image and detection results
    #cv2.imshow("Original", image)
    #cv2.imshow("Pests Detected", cv2.bitwise_and(image, image, mask=mask))
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    return result
