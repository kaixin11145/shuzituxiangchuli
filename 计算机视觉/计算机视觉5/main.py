import cv2

# 加载预训练的人脸检测分类器（OpenCV自带的Haar特征分类器）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# 加载预训练的人眼检测分类器（OpenCV自带的Haar特征分类器）
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# 读取图像，这里要替换成你实际的图像路径和文件名哦
image_path = "1.png"
img = cv2.imread(image_path)

# 将图像转换为灰度图像，因为人脸检测分类器通常在灰度图像上操作
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行人脸检测
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 在检测到的人脸位置绘制矩形框标记出来（使用蓝色，即(255, 0, 0)，线宽为2）
for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    # 获取当前人脸区域对应的灰度图和彩色图，用于人眼检测
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]
    # 在每个人脸区域内进行人眼检测
    eyes = eye_cascade.detectMultiScale(roi_gray)
    # 在检测到的人眼位置绘制矩形框标记出来（使用绿色，即(0, 255, 0)，线宽为2）
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

# 显示带有检测结果的图像
cv2.imshow('Viola-Jones Face and Eye Detection Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()