import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载图像
image = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)

# 使用Otsu方法直接进行图像分割
_, otsu_threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 方法a: 计算梯度幅度
sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
gradient_magnitude = np.uint8(gradient_magnitude)

# 方法b: 指定一个阈值T
T = 50  # 这个阈值可以根据实际情况调整

# 方法c: 使用阈值T对梯度幅度图像进行阈值处理,产生gτ
_, thresholded_gradient = cv2.threshold(gradient_magnitude, T, 255, cv2.THRESH_BINARY)

# 方法d: 仅使用对应于gτ(x,y)像素值为1的位置的像素计算直方图
mask = thresholded_gradient == 255
masked_image = np.where(mask, image, 0)
# 计算掩模图像的直方图
hist_masked_image = cv2.calcHist([image], [0], mask.astype(np.uint8), [256], [0, 256])

# 方法e: 使用Otsu方法全局地分割f(x,y)
_, otsu_threshold_masked = cv2.threshold(masked_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 使用matplotlib显示所有结果
plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title('Otsu Thresholding')
plt.imshow(otsu_threshold, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title('Gradient Magnitude')
plt.imshow(gradient_magnitude, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title('Gradient Thresholding')
plt.imshow(thresholded_gradient, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title('Masked Image')
plt.imshow(masked_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title('Otsu on Masked Image')
plt.imshow(otsu_threshold_masked, cmap='gray')
plt.axis('off')

# 显示直方图
plt.figure(figsize=(10, 4))
plt.plot(hist_masked_image, color='red', label='Masked Image Histogram')
plt.xlim([0, 256])
plt.legend()
plt.show()