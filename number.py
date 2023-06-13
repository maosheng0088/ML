from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os
import sys
# 加载保存的模型


model_path = 'C:/Users/32062/Desktop/new/cla/model.h5'
model = load_model(model_path)
# 使用模型进行预测

image = cv2.imread('number.png', cv2.IMREAD_GRAYSCALE)
resized_image = cv2.resize(image, (28, 28))
normalized_image = resized_image / 255.0
input_image = np.reshape(normalized_image, (1, 28, 28, 1))

# 使用模型进行预测
predictions = model.predict(input_image)

# 获取预测结果
predicted_label = np.argmax(predictions)

print('Predicted label:', predicted_label)
