import cv2
import numpy as np
import matplotlib.pyplot as plt


# 灰度级切片函数
def gray_level_slicing(image, lower_bound, upper_bound):
    result = image.copy()
    result[(image >= lower_bound) & (image <= upper_bound)] = 255
    result[(image < lower_bound) | (image > upper_bound)] = 0
    return result


# 位平面切片函数
def bit_plane_slicing(image):
    bit_planes = []
    for i in range(8):
        bit_plane = np.bitwise_and(image, 2 ** i)
        bit_planes.append(bit_plane)
    return bit_planes


# 直方图统计函数
def histogram(image):
    hist = np.zeros(256)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            hist[image[i, j]] += 1
    return hist


# 直方图均衡化函数
def histogram_equalization(image):
    hist = histogram(image)
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    image_equalized = np.interp(image.flatten(), np.arange(0, 256), cdf_normalized).reshape(image.shape)
    return image_equalized.astype(np.uint8)


# 均值滤波器函数
def mean_filter(image, kernel_size):
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image


# 方框滤波器函数
def box_filter(image, kernel_size):
    filtered_image = cv2.boxFilter(image, -1, (kernel_size, kernel_size))
    return filtered_image


# 高斯滤波器函数
def gaussian_filter(image, kernel_size, sigma):
    filtered_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered_image


# 读取灰度图像
image_path = 'ba7dd1afe95c58ce8e9cec84ce28b276.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

while True:
    print("图像处理系统菜单：")
    print("1. 灰度级切片")
    print("2. 位平面切片")
    print("3. 显示原始图像直方图")
    print("4. 显示均衡化后图像直方图")
    print("5. 显示三种滤波器处理前后对比")
    print("6. 退出")
    choice = input("请输入你的选择：")

    if choice == "1":
        lower_bound = 50
        upper_bound = 150
        sliced_image = gray_level_slicing(image, lower_bound, upper_bound)
        def slicing_transform(r):
            return 255 if lower_bound <= r <= upper_bound else 0
        r = np.arange(256)
        t = np.array([slicing_transform(i) for i in r])
        plt.figure(figsize=(15, 5))
        plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('Original Image')
        plt.subplot(132), plt.imshow(sliced_image, cmap='gray'), plt.title('Gray Level Sliced Image')
        plt.subplot(133), plt.plot(r, t), plt.title('Transformation Function T(r)')
        plt.show()
    elif choice == "2":
        bit_planes = bit_plane_slicing(image)
        plt.figure(figsize=(12, 4))
        for i in range(8):
            plt.subplot(2, 4, i + 1), plt.imshow(bit_planes[i], cmap='gray'), plt.title(f'Bit Plane {i} Image')
        plt.show()
    elif choice == "3":
        hist = histogram(image)
        plt.figure()
        plt.bar(range(256), hist)
        plt.xlabel('Gray Level')
        plt.ylabel('Frequency')
        plt.title('Histogram of Original Image')
        plt.show()
    elif choice == "4":
        equalized_image = histogram_equalization(image)
        equalized_hist = histogram(equalized_image)
        plt.figure()
        plt.bar(range(256), equalized_hist)
        plt.xlabel('Gray Level')
        plt.ylabel('Frequency')
        plt.title('Histogram of Equalized Image')
        plt.show()
    elif choice == "5":
        mean_filtered = mean_filter(image, 5)
        box_filtered = box_filter(image, 5)
        gaussian_filtered = gaussian_filter(image, 5, 1.4)
        plt.figure(figsize=(15, 5))
        plt.subplot(141), plt.imshow(image, cmap='gray'), plt.title('Original Image')
        plt.subplot(142), plt.imshow(mean_filtered, cmap='gray'), plt.title('Mean Filtered Image')
        plt.subplot(143), plt.imshow(box_filtered, cmap='gray'), plt.title('Box Filtered Image')
        plt.subplot(144), plt.imshow(gaussian_filtered, cmap='gray'), plt.title('Gaussian Filtered Image')
        plt.show()
        mean_diff = np.sum(np.abs(image - mean_filtered))
        box_diff = np.sum(np.abs(image - box_filtered))
        gaussian_diff = np.sum(np.abs(image - gaussian_filtered))
        if mean_diff < box_diff and mean_diff < gaussian_diff:
            print("均值滤波器在保留图像特征方面相对较好。")
        elif box_diff < mean_diff and box_diff < gaussian_diff:
            print("方框滤波器在保留图像特征方面相对较好。")
        else:
            print("高斯滤波器在保留图像特征方面相对较好。")
    elif choice == "6":
        break
    else:
        print("无效选择。")