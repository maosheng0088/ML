import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# 1. 数据准备
# 加载MNIST数据集
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据预处理
x_train = x_train.reshape(-1, 28, 28, 1)  # 调整输入图像的形状
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.astype('float32') / 255  # 归一化
x_test = x_test.astype('float32') / 255

y_train = to_categorical(y_train)  # 将标签转换为独热编码
y_test = to_categorical(y_test)

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

# 3. 编译模型
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# 4. 模型训练
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# 5. 模型评估
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('测试集准确率:', accuracy)
