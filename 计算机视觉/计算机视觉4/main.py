import cv2 as cv
import numpy as np

# 读图
img = cv.imread('1.jpg')
# 灰度化
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
# 提取canny边缘
edges = cv.Canny(gray, 50, 150, apertureSize=3)
# 霍夫变换
lines = cv.HoughLines(edges, 1, np.pi / 180, 200)
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    # 画线
    cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

cv.imwrite('houghlines.jpg', img)