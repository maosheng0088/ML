from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import sys
# 加载保存的模型


model_path = 'C:/Users/32062/Desktop/new/cla/garbage_classification_model.h5'
model = load_model(model_path)
# 使用模型进行预测

image_path = 'R-C.jpg'
image = cv2.imread(image_path)
image = cv2.resize(image, (64, 64))
image = image / 255.0
image = np.expand_dims(image, axis=0)
prediction = model.predict(image)
predicted_class = np.argmax(prediction)

# 获取类别标签
categories = ['garbageBagImages', 'paperBagImages']
predicted_label = categories[predicted_class]

print('预测结果:', predicted_label)
