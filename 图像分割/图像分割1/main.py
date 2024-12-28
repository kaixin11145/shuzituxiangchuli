import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

# 高斯平滑处理 不需要 效果不明显
smoothed_image = cv2.GaussianBlur(image, (3, 3), 0)

# 定义Prewitt算子
kernelx = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)
kernely = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=int)

# 对角线Prewitt算子
kernelx_diag1 = np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]], dtype=int)
kernely_diag1 = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]], dtype=int)

# 应用Prewitt算子（平滑处理）
x_smoothed = cv2.filter2D(smoothed_image, cv2.CV_32F, kernelx)
y_smoothed = cv2.filter2D(smoothed_image, cv2.CV_32F, kernely)

# 应用Prewitt算子（不平滑处理）
x = cv2.filter2D(image, cv2.CV_32F, kernelx)
y = cv2.filter2D(image, cv2.CV_32F, kernely)

# 应用对角线Prewitt算子
x_diag1 = cv2.filter2D(image, cv2.CV_32F, kernelx_diag1)
y_diag1 = cv2.filter2D(image, cv2.CV_32F, kernely_diag1)

# 计算梯度幅值（平滑处理）
absX_smoothed = cv2.convertScaleAbs(x_smoothed)
absY_smoothed = cv2.convertScaleAbs(y_smoothed)
Prewitt_smoothed = cv2.addWeighted(absX_smoothed, 0.5, absY_smoothed, 0.5, 0)

# 计算梯度幅值（不平滑处理）
absX = cv2.convertScaleAbs(x)
absY = cv2.convertScaleAbs(y)
Prewitt = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

# 计算对角线梯度幅值
absX_diag1 = cv2.convertScaleAbs(x_diag1)
absY_diag1 = cv2.convertScaleAbs(y_diag1)
Prewitt_diag1 = cv2.addWeighted(absX_diag1, 0.5, absY_diag1, 0.5, 0)

# 阈值化处理
_, thresholded = cv2.threshold(Prewitt, 50, 255, cv2.THRESH_BINARY)
_, thresholded_smoothed = cv2.threshold(Prewitt_smoothed, 50, 255, cv2.THRESH_BINARY)
_, thresholded_diag1 = cv2.threshold(Prewitt_diag1, 50, 255, cv2.THRESH_BINARY)

# 使用matplotlib显示图像
plt.figure(figsize=(12, 10))

# 显示原图
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# 显示不平滑处理的Prewitt边缘检测结果
plt.subplot(2, 3, 5)
plt.imshow(Prewitt, cmap='gray')
plt.title('Prewitt (No Smoothing)')
plt.axis('off')

# 显示平滑处理的Prewitt边缘检测结果
plt.subplot(2, 3, 4)
plt.imshow(Prewitt_smoothed, cmap='gray')
plt.title('Prewitt (With Smoothing)')
plt.axis('off')


# 显示对角线Prewitt边缘检测结果 不需要 效果不明显
plt.subplot(2, 3, 3)
plt.imshow(Prewitt_diag1, cmap='gray')
plt.title('Prewitt Diagonal')
plt.axis('off')

# 显示阈值化边缘检测
plt.subplot(2, 3, 2)
plt.imshow(thresholded, cmap='gray')
plt.title('Thresholded Edges')
plt.axis('off')

plt.tight_layout()
plt.show()