import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图片，这里以读入一张灰度图为例，你需要替换成自己正确的图片路径
image = cv2.imread('1.jpg', 0)  # 参数0表示以灰度图模式读取
print("图像读取后的数据类型:", image.dtype)
print("图像读取后的数据维度:", image.shape)

# 将图像转换为浮点型，这是Harris角点检测函数要求的数据类型
image_float = np.float32(image)
print("转换为float32后的数据类型:", image_float.dtype)
print("转换为float32后的数据维度:", image_float.shape)

# 进行Harris角点检测
# 下面几个参数解释：
# blockSize：角点检测中考虑的邻域大小，一般取奇数，例如2，3，5等，这里取2
# ksize：Sobel算子的孔径大小，用于计算图像梯度，一般取1，3，5等奇数，这里取3
# k：Harris角点检测的自由参数，取值通常在0.04 - 0.06之间，这里取0.04
dst = cv2.cornerHarris(image_float, blockSize=2, ksize=3, k=0.04)

# 对检测结果进行膨胀操作，以便标记出更明显的角点，这是一个可选的处理步骤
dst = cv2.dilate(dst, None)

# 设置一个阈值，用于筛选出比较显著的角点，这里阈值取0.01 * dst.max()
threshold = 0.01 * dst.max()
image_corners = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)  # 将灰度图转换为彩色图（BGR格式）

# 获取图像的高度和宽度
height, width = image.shape[:2]
for y in range(height):
    for x in range(width):
        if dst[y, x] > threshold:
            # 将角点位置的像素设置为蓝色（在BGR格式中，蓝色对应(255, 0, 0)）
            image_corners[y, x] = [255, 0, 0]

# 可视化原始图像和标记了角点的图像
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB), cmap='gray')  # 将灰度图转换为RGB格式用于matplotlib展示
plt.title('Original Image')

plt.subplot(1, 2, 2)
plt.imshow(cv2.cvtColor(image_corners, cv2.COLOR_BGR2RGB), cmap='gray')  # 将标记角点后的BGR图像转换为RGB格式展示
plt.title('Image with Harris Corners in Blue')

plt.show()