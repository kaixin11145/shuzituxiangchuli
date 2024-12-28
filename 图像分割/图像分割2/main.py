import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取图像
image = cv2.imread('img_1.png', cv2.IMREAD_GRAYSCALE)  # 确保路径正确

# 使用Canny算子进行边缘检测
canny_edges = cv2.Canny(image, 100, 200)

# 使用Prewitt算子进行边缘检测
kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
kernely = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
prewitt_x = cv2.filter2D(image, -1, kernelx)
prewitt_y = cv2.filter2D(image, -1, kernely)
prewitt_edges = cv2.addWeighted(prewitt_x, 0.5, prewitt_y, 0.5, 0)

# 显示结果
plt.figure(figsize=(12, 8))

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])

plt.subplot(132)
plt.imshow(canny_edges, cmap='gray')
plt.title('Canny Edge Detection'), plt.xticks([]), plt.yticks([])

plt.subplot(133)
plt.imshow(prewitt_edges, cmap='gray')
plt.title('Prewitt Edge Detection'), plt.xticks([]), plt.yticks([])

plt.show()