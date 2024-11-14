import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image_path):
    # Read the image
    img = cv2.imread(image_path)  # Use function parameter
    if img is None:
        print("Image file not found")
        return None

    # Resize the image
    img_resized = cv2.resize(img, (500, 500))  # Resize image to 500x500

    # Convert to grayscale
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur for noise reduction
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # Display original and preprocessed images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')

    plt.subplot(1, 3, 2)
    plt.imshow(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    plt.title('Resized Image')

    plt.subplot(1, 3, 3)
    plt.imshow(img_blurred, cmap='gray')
    plt.title('Blurred and Grayscale Image')
    plt.show()

    return img_blurred  # Return preprocessed image

def extract_features_from_image(img, img_blurred):
    # Ensure image is not None
    if img_blurred is None:
        print("Image data not provided")
        return

    # Edge detection - using Canny algorithm
    edges = cv2.Canny(img_blurred, 100, 200)

    # Texture analysis - using Laplacian algorithm
    laplacian = cv2.Laplacian(img_blurred, cv2.CV_64F)

    # Color histogram
    color = ('b', 'g', 'r')
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    # Display results
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(132), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
    plt.subplot(133), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
    plt.show()

# Usage example
img_blurred = preprocess_image('apple1.png')
extract_features_from_image(cv2.imread('apple1.png'), img_blurred)
