import cv2
import numpy as np
from matplotlib import pyplot as plt

def preprocess_image(image_path):
    # 读取图像
    img = cv2.imread(image_path)  # 使用函数参数
    if img is None:
        print("图像文件未找到")
        return None

    # 调整图像大小
    img_resized = cv2.resize(img, (500, 500))  # 将图像大小调整为500x500

    # 转换为灰度图像
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

    # 应用高斯滤波去噪
    img_blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)

    # 显示原始图像和预处理后的图像
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

    return img_blurred  # 返回预处理后的图像

def extract_features_from_image(img, img_blurred):
    # 确保图像不是None
    if img_blurred is None:
        print("图像数据未提供")
        return

    # 边缘检测 - 使用Canny算法
    edges = cv2.Canny(img_blurred, 100, 200)

    # 纹理分析 - 使用Laplacian算法
    laplacian = cv2.Laplacian(img_blurred, cv2.CV_64F)

    # 颜色直方图
    color = ('b', 'g', 'r')
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(histr, color=col)
        plt.xlim([0, 256])

    # 显示结果
    plt.figure(figsize=(15, 5))
    plt.subplot(131), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('Original')
    plt.subplot(132), plt.imshow(edges, cmap='gray'), plt.title('Canny Edges')
    plt.subplot(133), plt.imshow(laplacian, cmap='gray'), plt.title('Laplacian')
    plt.show()

# 使用示例
img_blurred = preprocess_image('apple1.png')
extract_features_from_image(cv2.imread('apple1.png'), img_blurred)

