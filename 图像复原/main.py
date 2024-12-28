import cv2
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 读取灰度图像
image = cv2.imread('img_2.png', cv2.IMREAD_GRAYSCALE)


# 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=10):
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image + gauss, 0, 255)
    return noisy


# 添加均匀噪声
def add_uniform_noise(image, low=0, high=50):
    uniform = np.random.uniform(low, high, image.shape)
    noisy = np.clip(image + uniform, 0, 255)
    return noisy


# 添加椒盐噪声
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy = np.copy(image)
    total_pixels = image.size
    num_salt = np.ceil(salt_prob * total_pixels)
    num_pepper = np.ceil(pepper_prob * total_pixels)

    # 添加盐噪声
    coords_y_salt = np.random.randint(0, image.shape[0], int(num_salt))
    coords_x_salt = np.random.randint(0, image.shape[1], int(num_salt))
    noisy[coords_y_salt, coords_x_salt] = 255

    # 添加椒噪声
    coords_y_pepper = np.random.randint(0, image.shape[0], int(num_pepper))
    coords_x_pepper = np.random.randint(0, image.shape[1], int(num_pepper))
    noisy[coords_y_pepper, coords_x_pepper] = 0

    return noisy


# 应用噪声
gaussian_noisy_image = add_gaussian_noise(image)
uniform_noisy_image = add_uniform_noise(image)
salt_and_pepper_noisy_image = add_salt_and_pepper_noise(image)


# 绘制图像及其直方图
def plot_images_and_hist(images, titles):
    plt.figure(figsize=(12, 6))
    for i in range(len(images)):
        plt.subplot(2, len(images), i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')

        plt.subplot(2, len(images), i + 1 + len(images))
        plt.hist(images[i].ravel(), bins=256, range=[0, 256], histtype='step')
        plt.title('Histogram of ' + titles[i])

    plt.tight_layout()
    plt.show()


plot_images_and_hist(
    [image, gaussian_noisy_image, uniform_noisy_image, salt_and_pepper_noisy_image],
    ['Original Image', 'Gaussian Noisy Image', 'Uniform Noisy Image', 'Salt and Pepper Noisy Image']
)


# 中值滤波
def apply_median_filter(image):
    return cv2.medianBlur(image.astype(np.uint8), 5)


# 高斯滤波
def apply_gaussian_filter(image):
    return cv2.GaussianBlur(image.astype(np.uint8), (5, 5), 0)


# 对每种噪声图像应用滤波器
gaussian_noisy_image = add_gaussian_noise(image)
uniform_noisy_image = add_uniform_noise(image)
salt_and_pepper_noisy_image = add_salt_and_pepper_noise(image)

# 应用最合适的滤波器
filtered_gaussian_noisy_image = apply_gaussian_filter(gaussian_noisy_image)
filtered_uniform_noisy_image = apply_median_filter(uniform_noisy_image)
filtered_salt_and_pepper_noisy_image = apply_median_filter(salt_and_pepper_noisy_image)

# 绘制滤波前后的对比图
plot_images_and_hist(
    [
        gaussian_noisy_image, filtered_gaussian_noisy_image
    ],['Gaussian Noisy Image', 'Filtered Gaussian']
)
plot_images_and_hist(
    [
        uniform_noisy_image, filtered_uniform_noisy_image
    ], ['Uniform Noisy Image', 'Filtered Uniform']
)
plot_images_and_hist(
    [
        salt_and_pepper_noisy_image, filtered_salt_and_pepper_noisy_image
    ], ['Salt and Pepper Noisy Image', 'Filtered Salt and Pepper'
    ]
)


# 添加运动模糊
def add_motion_blur(image, size=15):
    kernel = np.zeros((size, size))
    kernel[int((size - 1) / 2), :] = np.ones(size)
    kernel /= size
    blurred = cv2.filter2D(image, -1, kernel)
    return blurred

# 添加高斯噪声
def add_gaussian_noise(image, mean=0, var=0.001):
    row, col = image.shape
    sigma = var ** 0.5
    gauss = np.random.normal(mean, sigma, (row, col))
    noisy = image + gauss
    return noisy

# 维纳滤波
def wiener_filter(image, kernel, K=0.01):
    f_image = np.fft.fft2(image)
    f_kernel = np.fft.fft2(kernel, s=image.shape)
    conj_kernel = np.conj(f_kernel)
    H = conj_kernel / (np.abs(f_kernel) ** 2 + K)
    f_image_filtered = f_image * H
    image_filtered = np.fft.ifft2(f_image_filtered)
    image_filtered = np.real(image_filtered)
    image_filtered = np.clip(image_filtered, 0, 255).astype(np.uint8)
    return image_filtered

# CLS滤波
def constrained_least_squares_filter(noisy_image, kernel, lambda_value=0.01):
    dft_noisy_image = np.fft.fft2(noisy_image)
    dft_kernel = np.fft.fft2(kernel, s=noisy_image.shape)
    conj_kernel = np.conjugate(dft_kernel)
    kernel_mag_squared = np.abs(dft_kernel) ** 2
    numerator = conj_kernel
    denominator = kernel_mag_squared + lambda_value
    filter_response = numerator / denominator
    restored_image_fft = dft_noisy_image * filter_response
    restored_image = np.fft.ifft2(restored_image_fft)
    restored_image = np.abs(restored_image)
    return np.clip(restored_image, 0, 255).astype(np.uint8)

# 绘制图像及其直方图
def plot_images_and_hist(images, titles):
    plt.figure(figsize=(15, 5))
    for i, (image, title) in enumerate(zip(images, titles), 1):
        plt.subplot(1, len(images), i)
        plt.imshow(image, cmap='gray')
        plt.title(titles[i-1])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# 使用示例
motion_blurred_image = add_motion_blur(image)
motion_blurred_noisy_image = add_gaussian_noise(motion_blurred_image, var=0.01)

motion_kernel = np.zeros((15, 15))
motion_kernel[int((15 - 1) / 2), :] = np.ones(15)
motion_kernel /= 15

# 调整维纳滤波参数
K = 0.01  # 维纳滤波的噪声功率谱估计值
wiener_restored = wiener_filter(motion_blurred_noisy_image, motion_kernel, K)

# 调整CLS滤波参数
lambda_value = 0.01  # CLS滤波的正则化参数
cls_restored = constrained_least_squares_filter(motion_blurred_noisy_image, motion_kernel, lambda_value)

plot_images_and_hist(
    [motion_blurred_noisy_image, wiener_restored, cls_restored],
    ['Motion Blurred Noisy', 'Wiener Restored', 'CLS Restored']
)