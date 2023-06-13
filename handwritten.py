from tensorflow.keras.datasets import mnist
# 下载数据集
# x_train_original和y_train_original代表训练集的图像与标签, x_test_original与y_test_original代表测试集的图像与标签
(x_train_original, y_train_original), (x_test_original, y_test_original) = mnist.load_data()
import matplotlib.pyplot as plt
def mnist_visualize_multiple(start,end,length,width):
    for i in range(start,end):
        plt.subplot(length,width,i+1)
        plt.imshow(x_train_original[i],cmap=plt.get_cmap('gray'))
        title = 'label='+ str(y_train_original[i])
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show()
mnist_visualize_multiple(start=0,end=6,length=3,width=2)

print('训练集图像的尺寸：',x_train_original.shape)
print('训练集标签的尺寸：',y_train_original.shape)
print('测试集图像的尺寸：',x_test_original.shape)
print('测试集标签的尺寸：',y_test_original.shape)
x_val = x_train_original[50000:]
y_val = y_train_original[50000:]
x_train = x_train_original[:50000]
y_train = y_train_original[:50000]
# 打印验证集数据量
print('验证集图像的尺寸：', x_val.shape)
print('验证集标签的尺寸：', y_val.shape)
print('训练集图像的尺寸：',x_train.shape)
print('训练集标签的尺寸：',y_train.shape)

x_train = x_train.reshape(x_train.shape[0],28,28,1).astype('float32')
x_val = x_val.reshape(x_val.shape[0],28,28,1).astype('float32')
x_test = x_test_original.reshape(x_test_original.shape[0],28,28,1).astype('float32')

x_train = x_train/255
x_test = x_test/255
x_val = x_val/255

print('训练集传入网络的图像尺寸：', x_train.shape)
print('验证集传入网络的图像尺寸：', x_val.shape)
print('测试集传入网络的图像尺寸：', x_test.shape)

import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense


def CNN_model():
    model = Sequential()
    model.add(Conv2D(filters=6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=16, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())

    model.add(Dense(120, activation='relu'))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model


model = CNN_model()
print(model.summary())
import tensorflow as tf
model.compile(optimizer='adam',metrics=['accuracy'],loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
train_history = model.fit(x_train,y_train,validation_data=(x_val,y_val),epochs=5,batch_size=32)
model.save('model.h5')

import pickle
with open('train_history.txt','wb') as file_pi:
    pickle.dump(train_history.history,file_pi)
with open('train_history.txt','rb') as file_pi:
    history=pickle.load(file_pi)
def show_train_history(history, train, validation):
    plt.plot(history[train])
    plt.plot(history[validation])
    plt.title('Train History')
    plt.xlabel('epoch')
    plt.ylabel(train)
    plt.legend(['train','validation'],loc='best')
    plt.show()


# 准确率
show_train_history(history, 'accuracy', 'val_accuracy')
show_train_history(history, 'loss', 'val_loss')
score = model.evaluate(x_test,y_test_original)
# 测试集上的损失与精度信息
print('test loss',score[0])
print('test accuracy',score[1])
preditions = model.predict(x_test)
preditions = np.argmax(preditions,axis=1)
print('前20张图片预测结果：',preditions[:20])
def mnist_visualize_multiple_predict(start, end, length, width):
    plt.figure(figsize=(9,9))
    for i in range(start,end):
        plt.subplot(length,width,i+1)
        plt.imshow(x_test_original[i],cmap=plt.get_cmap('gray'))
        title_true = 'true:'+ str(y_test_original[i])
        title_pre = 'pred:'+str(preditions[i])
        title = title_true + ';' + title_pre
        plt.title(title)
        plt.xticks([])
        plt.yticks([])
    plt.show()
mnist_visualize_multiple_predict(start=0, end=9, length=3, width=3)
from sklearn.metrics import confusion_matrix
class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
cm = confusion_matrix(y_test_original, preditions,labels=[0,1,2,3,4,5,6,7,8,9])
cm = np.array(cm)
def plot_confusion_matrix(cm):
    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.annotate(cm[j, i], xy=(i, j), horizontalalignment='center', verticalalignment='center')
    plt.imshow(cm, interpolation='nearest',cmap=plt.cm.Blues)  # 在特定的窗口上显示图像
    plt.title("Confusion Matrix")  # 图像标题
    class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    num_local = np.array(range(len(class_names)))
    plt.xticks(num_local, class_names, fontsize=15, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, class_names, fontsize=15, rotation=0)  # 将标签印在y轴坐标上
    plt.xlabel("Predicted")
    plt.ylabel("Actual")


plt.figure(figsize=(10,10))
plot_confusion_matrix(cm)