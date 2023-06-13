import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 数据集目录
dataset_dir = ''

# 类别列表
categories = ['garbageBagImages', 'paperBagImages']

# 准备图像数据和标签
images = []
labels = []

for category_id, category in enumerate(categories):
    category_dir = os.path.join(dataset_dir, category)
    for image_name in os.listdir(category_dir):
        image_path = os.path.join(category_dir, image_name)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (64, 64))  # 调整图像大小
        images.append(image)
        labels.append(category_id)

# 转换为NumPy数组
images = np.array(images)
labels = np.array(labels)

# 对图像数据进行归一化
images = images / 255.0

# 进行独热编码
num_classes = len(categories)
labels = tf.keras.utils.to_categorical(labels, num_classes)

train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# 1. 搭建神经网络模型
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, kernel_size=(3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(2, activation='softmax')
])

# 2. 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=10, batch_size=32)

# 4. 模型评估
loss, accuracy = model.evaluate(test_images, test_labels)
print('测试集准确率:', accuracy)
model.save('garbage_classification_model.h5')

