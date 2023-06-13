test_img_path = ["./638201107972830000.jpg"]


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread(test_img_path[0])

# 展示待预测图片
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis('off')
plt.show()

import paddlehub as hub

module = hub.Module(name="pyramidbox_lite_mobile_mask")

import os

import cv2

imgs = [cv2.imread(test_img_path[0])]

# 口罩检测预测
# visualization=True 将预测结果保存图片可视化
# output_dir='detection_result' 预测结果图片保存在当前运行路径下detection_result文件夹下
results = module.face_detection(images=imgs, use_multi_scale=True, shrink=0.6, visualization=True, output_dir='detection_result')
for result in results:
    print(result)

