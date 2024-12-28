import numpy as np
import cv2
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 加载图片
img = cv2.imread('1.jpg', 0)  # 以灰度模式加载图片
img = img / 255.0  # 归一化

# 进行PCA降维
pca = PCA()
pca.fit(img)

# 选择特征值个数
cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
n_components = np.argmax(cumulative_variance >= 0.99) + 1  # 选择累积贡献率大于99%的特征值个数

# 对图片进行恢复
pca = PCA(n_components=n_components)
img_reduced = pca.fit_transform(img)
img_restored = pca.inverse_transform(img_reduced)

# 显示原始图片和恢复后的图片
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img, cmap='gray')
plt.text(0.5, -0.1, '(a)', transform=plt.gca().transAxes, fontsize=12, ha='center')  # 添加序号(a)


plt.subplot(1, 2, 2)
plt.title('Restored Image')
plt.imshow(img_restored, cmap='gray')
plt.text(0.5, -0.1, '(b)', transform=plt.gca().transAxes, fontsize=12, ha='center')  # 添加序号(b)

plt.show()
