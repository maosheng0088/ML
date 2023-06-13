import cv2
from matplotlib import pyplot as plt
img = cv2.imread('face.jpg')

# 转换为灰度图
gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_img,1.3,2)
for (x, y, w, h) in faces:
    # 在原图像上绘制矩形
    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.title('')
plt.show()
