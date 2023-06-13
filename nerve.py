from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, MaxPooling2D, Dense

# 第一个卷积层：使用6个5x5的卷积核，输入图像大小为32x32x3（RGB图像），使用valid填充方式（不进行边缘填充）。
# 第一个池化层：使用2x2的池化窗口进行最大池化。
# 第二个卷积层：使用16个5x5的卷积核，使用valid填充方式。
# 第二个池化层：使用2x2的池化窗口进行最大池化。
# Flatten层：将特征图展平为一维向量。
# 第一个全连接层：包含120个神经元，并使用ReLU激活函数。
# 第二个全连接层：包含84个神经元，并使用ReLU激活函数。
# 第三个全连接层（输出层）：包含10个神经元，对应于10个类别的softmax分类。
#

model = Sequential([
    Conv2D(filters = 6,kernel_size = 5,padding = 'valid',input_shape = (32,32,3)),
    MaxPooling2D(pool_size = 2),
    Conv2D(filters = 16,kernel_size = 5,padding = 'valid'),
    MaxPooling2D(pool_size = 2),
    Flatten(),
    Dense(120,activation='relu'),
    Dense(84,activation='relu'),
    Dense(10,activation='softmax'),
])

model.summary()