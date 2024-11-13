import cv2
import numpy as np

def get_hsv_values(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found or path is incorrect.")
        return

    # 将图像从BGR转换到HSV色彩空间
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 选择ROI (Region of Interest)
    roi = cv2.selectROI("Select ROI", hsv_image, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi
    if w == 0 or h == 0:
        print("Error: No region selected.")
        cv2.destroyAllWindows()
        return

    roi_cropped = hsv_image[y:y+h, x:x+w]

    # 计算选定区域的平均HSV值
    if roi_cropped.size == 0:
        print("Error: Selected region is empty.")
    else:
        average_hsv = np.mean(roi_cropped, axis=(0, 1))
        # 输出平均HSV值
        print("Average HSV values for the selected region:")
        print("H: {:.2f}, S: {:.2f}, V: {:.2f}".format(average_hsv[0], average_hsv[1], average_hsv[2]))

    # 关闭所有窗口
    cv2.destroyAllWindows()

# 使用示例
get_hsv_values("pear2.png")

