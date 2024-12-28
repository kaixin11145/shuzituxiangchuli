import cv2
import numpy as np
from matplotlib import pyplot as plt

# 读取彩色图像
img = cv2.imread('IMG_20241004_183027(1).jpg')

# RGB 转 HSI
def rgb2hsi(rgb_img):
    r, g, b = cv2.split(rgb_img)
    r, g, b = r / 255.0, g / 255.0, b / 255.0
    num = 0.5 * ((r - g) + (r - b))
    den = np.sqrt((r - g)**2 + (r - b) * (g - b))
    theta = np.arccos(num / (den + 1e-5))
    h = theta
    h[b > g] = 2 * np.pi - h[b > g]
    h = h / (2 * np.pi)
    s = 1 - 3 * np.minimum(r, g, b) / (r + g + b + 1e-5)
    i = (r + g + b) / 3
    return np.stack([h, s, i], axis=-1)

hsi_img = rgb2hsi(img)

# RGB 分量图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(img[:, :, 2], cmap='Reds')
axs[0].set_title('R')
axs[1].imshow(img[:, :, 1], cmap='Greens')
axs[1].set_title('G')
axs[2].imshow(img[:, :, 0], cmap='Blues')
axs[2].set_title('B')
plt.show()

# HSI 分量图
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(hsi_img[:, :, 0], cmap='hsv')
axs[0].set_title('H')
axs[1].imshow(hsi_img[:, :, 1], cmap='gray')
axs[1].set_title('S')
axs[2].imshow(hsi_img[:, :, 2], cmap='gray')
axs[2].set_title('I')
plt.show()


# RGB 直方图均衡化
for i in range(3):
    img[:, :, i] = cv2.equalizeHist(img[:, :, i])

# HSI 整体直方图均衡化
for j in range(3):
    hsi_img[:, :, j] = cv2.equalizeHist((hsi_img[:, :, j] * 255).astype(np.uint8))
    hsi_img[:, :, j] /= 255.0

# 绘制 RGB 直方图均衡化的结果
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    axs[i].hist(img[:, :, i].ravel(), 256, [0, 256])
    axs[i].set_title(f'RGB channel {i} after equalization')
plt.show()

# 绘制 HSI 整体直方图均衡化后的结果
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
for i in range(3):
    axs[i].hist(hsi_img[:, :, i].ravel(), 256, [0, 1])
    axs[i].set_title(f'HSI channel {i} after equalization')
plt.show()


# RGB 均值滤波
blur_rgb = cv2.blur(img, (5, 5))

# RGB 拉普拉斯变换
laplacian_rgb = cv2.Laplacian(img, cv2.CV_64F)

# HSI 强度分量均值滤波
hsi_img_blur = hsi_img.copy()
hsi_img_blur[:, :, 2] = cv2.blur(hsi_img_blur[:, :, 2], (5, 5))

# HSI 强度分量拉普拉斯变换
laplacian_hsi = cv2.Laplacian(hsi_img_blur[:, :, 2], cv2.CV_64F)

# 比较结果
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
axs[0, 0].imshow(img)
axs[0, 0].set_title('RGB after processing')
axs[0, 1].imshow(np.abs(laplacian_rgb), cmap='gray')
axs[0, 1].set_title('Laplacian on RGB')
axs[1, 0].imshow(hsi_img_blur)
axs[1, 0].set_title('HSI after processing')
axs[1, 1].imshow(np.abs(laplacian_hsi), cmap='gray')
axs[1, 1].set_title('Laplacian on HSI intensity')
plt.show()