import cv2
import numpy as np
import matplotlib.pyplot as plt


def apply_filter(image, kernel):
    return cv2.filter2D(image, -1, kernel)


# 一阶锐化算子
def roberts(image):
    kernel = np.array([[1, 0], [0, -1]]) + np.array([[0, 1], [-1, 0]])
    return apply_filter(image, kernel)


def sobel(image):
    kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    return apply_filter(image, kernel)


def prewitt(image):
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return apply_filter(image, kernel_x) + apply_filter(image, kernel_y)


def kirsch(image):
    kernels = [np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),
               np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),
               np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),
               np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),
               np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, -3]]),
               np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),
               np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),
               np.array([[5, -3, -3], [-3, 0, -3], [-3, -3, -3]])]

    context = np.array([apply_filter(image, k) for k in kernels])

    return np.max(context, axis=0)


# 二阶锐化算子：拉普拉斯算子
def laplacian(image):
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    return apply_filter(image, kernel)


# 读取图像
image = cv2.imread('img_1.png', cv2.IMREAD_GRAYSCALE)  # 替换为你的图像路径
if image is None:
    raise FileNotFoundError("Image not found.")

# 应用不同算子
roberts_image = roberts(image)
sobel_image = sobel(image)
prewitt_image = prewitt(image)
kirsch_image = kirsch(image)
laplacian_image = laplacian(image)

# 显示原图和处理后的图像
plt.figure(figsize=(10, 8))
plt.subplot(3, 2, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 2)
plt.title('Roberts')
plt.imshow(roberts_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 3)
plt.title('Sobel')
plt.imshow(sobel_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 4)
plt.title('Prewitt')
plt.imshow(prewitt_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 5)
plt.title('Kirsch')
plt.imshow(kirsch_image, cmap='gray')
plt.axis('off')

plt.subplot(3, 2, 6)
plt.title('Laplacian')
plt.imshow(laplacian_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()