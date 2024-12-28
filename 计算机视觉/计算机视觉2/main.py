import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import data, exposure
import skimage.io as io


def extract_and_visualize_hog(image_path):
    """
    函数功能：读取指定路径的图片，提取其HOG特征，并可视化HOG归一化之后的直方图
    参数：
    image_path：要处理的图片的路径
    """
    # 读取图片
    image = io.imread(image_path)

    # 提取HOG特征，对于彩色图像（多通道）指定channel_axis参数，这里假设是常见的RGB图像，通道轴为-1
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(3, 3), block_norm='L2-Hys',
                        visualize=True, channel_axis=-1)

    # 方式一：直接绘制HOG可视化图像（归一化后的一种直观展示，类似直方图分布的灰度图）
    # 对HOG可视化图像进行强度缩放，使其更便于可视化显示，将像素值范围调整到 (0, 10) 区间
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    plt.title('HOG Normalized Image (Rescaled)')


    # 方式二：通过统计特征向量各方向数据绘制常规直方图
    # 假设特征向量fd的形状是 (n_features,)，尝试将其按照方向维度重塑
    num_orientations = 9
    fd_reshaped = fd.reshape(-1, num_orientations)
    # 计算每个方向上的直方图统计（这里简单采用求和的方式，也可根据实际需求采用其他统计方式）
    histogram = fd_reshaped.sum(axis=0)
    plt.subplot(1, 2, 2)
    plt.bar(range(num_orientations), histogram)
    plt.xlabel('Orientation')
    plt.ylabel('Magnitude')
    plt.title('HOG Histogram (by Summing Features)')


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 替换为你自己的图片实际路径
    image_path = '1.jpg'
    extract_and_visualize_hog(image_path)